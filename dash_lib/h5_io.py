from pathlib import Path
from datetime import datetime
import logging
from collections import OrderedDict
from functools import partial, wraps
import asyncio
import numpy as np

import xarray as xr
import h5py
from concurrent.futures import ProcessPoolExecutor
from diskcache import Cache
from numpy import iterable
import pandas as pd
import sys

logger = logging.getLogger(__name__)

cache = Cache(f"{Path.cwd()}/.cache/h5_metadata")

"""
Asynchronous HDF5 Processing with Process-Based Caching

This module provides a decorator-based approach for asynchronous processing of HDF5 files
with automatic caching. The architecture follows this flow:
1. Main thread coordinates async task execution
2. Each async task checks the cache for existing results
3. On cache miss, spawns a separate process to read HDF5 file (avoiding HDF5 threading issues)
4. Results are cached and returned to the main thread
5. All file processing is managed asynchronously but can run concurrently or sequentially
"""


def clear_cache():
    cache.clear()


# @cache.memoize()
def _execute_in_process(fn, *args, **kwargs):
    with ProcessPoolExecutor() as executor:
        future = executor.submit(fn, *args, **kwargs)
        return future.result()


def async_h5_processor_cache(func, use_cache=True):
    @wraps(func)
    def wrapper(h5_paths, *args, **kwargs):
        if isinstance(h5_paths, (str, Path)):
            _h5_paths = [h5_paths]
        else:
            _h5_paths = h5_paths

        if use_cache:
            process_executor = cache.memoize()(_execute_in_process)
        else:
            process_executor = _execute_in_process

        process_task = partial(process_executor, func, *args, **kwargs)

        # Use asyncio to coordinate multiple file processing tasks
        async def process_files():
            loop = asyncio.get_event_loop()
            # Create async tasks that will run the cached processor in the thread pool
            tasks = [
                loop.run_in_executor(None, process_task, path) for path in _h5_paths
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(process_files())
        return results if iterable(h5_paths) else results[0]

    return wrapper


def h5group_to_dict(h5_group):
    """
    Recursively converts an h5py Group and its contents into a Python dictionary.
    Datasets are loaded into memory as NumPy arrays.
    """
    result_dict = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            # If it's a subgroup, recursively call the function
            result_dict[key] = h5group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # If it's a dataset, load its data into a NumPy array
            result_dict[key] = item[()]  # [()] loads the entire dataset into memory
        # You can add handling for other h5py types if needed, e.g., external links
    return result_dict


def _load_metedata(file_path):
    file_path = Path(file_path)
    metadata = OrderedDict()
    metadata["filename"] = file_path.name
    metadata["filepath"] = str(file_path)

    metadata_paths = [
        {"key": "measurement_name", "path": "/measurement/spec_run", "attr": "name"},
        {"key": "sample", "path": "/app/settings", "attr": "sample"},
        {
            "key": "session",
            "path": "/hardware/mf_crucible_spinbot/settings",
            "attr": "session",
        },
        {
            "key": "user_name",
            "path": "/hardware/mf_crucible_spinbot/settings",
            "attr": "user_name",
        },
        {
            "key": "timestamp",
            "path": "/",
            "attr": "time_id",
            "transform": lambda x: datetime.fromtimestamp(x).isoformat(
                timespec="seconds"
            ),
        },
    ]

    with h5py.File(file_path, "r") as hf:
        for item in metadata_paths:
            try:
                attr_name = item["attr"] if "attr" in item else item["key"]
                value = hf[item["path"]].attrs[attr_name]
                if "transform" in item and callable(item["transform"]):
                    value = item["transform"](value)
                metadata[item["key"]] = value
            except Exception as e:
                logger.warning(
                    f"Warning: Failed to retrieve {item['key']} from {file_path}: {e}"
                )

        for k, v in hf["/process_params"].attrs.items():
            metadata[k] = v

    return metadata


load_metedata = async_h5_processor_cache(_load_metedata)


def _load_cd_data(file_path):
    with h5py.File(file_path, "r") as hf:
        cd_data = {
            "wavelength": hf["/results/cd/wavelength"][:],
            "front_CD": hf["/results/cd/front_CD"][:],
            "back_CD": hf["/results/cd/back_CD"][:],
            "front_abs": hf["/results/cd/front_abs"][:],
            "back_abs": hf["/results/cd/back_abs"][:],
            "gen_CD": hf["/results/cd/gen_CD"][:],
            "ldlb_CD": hf["/results/cd/ldlb_CD"][:],
            "abs_avg": hf["/results/cd/abs_avg"][:],
            "g_factor": hf["/results/cd/g_factor"][:],
        }

    return pd.DataFrame(cd_data)


load_cd_data = async_h5_processor_cache(_load_cd_data)


def _load_PL_image(file_path):
    with h5py.File(file_path, "r") as hf:
        pl_image = hf["/results/cbox/photo/image"][:]
    return pl_image


load_PL_image = async_h5_processor_cache(_load_PL_image)


def _load_metrics(file_path):
    with h5py.File(file_path, "r") as hf:
        metrics = h5group_to_dict(hf["/metrics"])

    return metrics


load_metrics = async_h5_processor_cache(_load_metrics)


def _load_xrd_int(file_path):
    with h5py.File(file_path, "r") as hf:
        xrd_int = {
            "2theta": hf["/results/xrd/2theta_Cu"][:],
            "intensity": hf["/results/xrd/int"][:],
        }

    return pd.DataFrame(xrd_int)


load_xrd_int = async_h5_processor_cache(_load_xrd_int)


def build_datasets(h5_file):
    with h5py.File(h5_file, "r") as hf:
        process_params_key = [
            "annealing_temp",
            "annealing_time",
            "spin_speed",
            "dispense_speed",
            "dispense_volume",
            "concentration",
        ]

        process_params_coords = {}

        if "process_params" in hf:
            process_params_group = hf["/process_params"]
            process_params_coords = dict(
                [k, (["process_params"], np.array([process_params_group.attrs[k]]))]
                for k in process_params_key
            )

        if "/results/cd" in hf:
            cd_data = {
                "wavelength": hf["/results/cd/wavelength"][:],
                "front_CD": hf["/results/cd/front_CD"][:],
                "back_CD": hf["/results/cd/back_CD"][:],
                "front_abs": hf["/results/cd/front_abs"][:],
                "back_abs": hf["/results/cd/back_abs"][:],
                "gen_CD": hf["/results/cd/gen_CD"][:],
                "ldlb_CD": hf["/results/cd/ldlb_CD"][:],
                "abs_avg": hf["/results/cd/abs_avg"][:],
                "g_factor": hf["/results/cd/g_factor"][:],
            }

            cd_dataset = xr.Dataset(
                {
                    "front_CD": (["wavelength"], cd_data["front_CD"]),
                    "back_CD": (["wavelength"], cd_data["back_CD"]),
                    "front_abs": (["wavelength"], cd_data["front_abs"]),
                    "back_abs": (["wavelength"], cd_data["back_abs"]),
                    "gen_CD": (["wavelength"], cd_data["gen_CD"]),
                    "ldlb_CD": (["wavelength"], cd_data["ldlb_CD"]),
                    "abs_avg": (["wavelength"], cd_data["abs_avg"]),
                    "g_factor": (["wavelength"], cd_data["g_factor"]),
                }
            )

            cd_dataset = cd_dataset.assign_coords({"wavelength": cd_data["wavelength"]})

            if process_params_coords:
                cd_dataset = cd_dataset.expand_dims("process_params", axis=0)
                cd_dataset = cd_dataset.assign_coords(process_params_coords)

        if "/results/cbox/photo" in hf:
            pl_img_data = {"pl_img": hf["/results/cbox/photo/image"][:]}

            pl_img_dataset = xr.Dataset(
                {"pl_img": (["h", "w", "c"], pl_img_data["pl_img"])}
            )

            if process_params_coords:
                pl_img_dataset = pl_img_dataset.expand_dims("process_params", axis=0)
                pl_img_dataset = pl_img_dataset.assign_coords(process_params_coords)

        if "/results/cbox/pl" in hf:
            pl_spectra_data = {
                "wavelength": hf["/results/cbox/pl/wavelength"][:],
                "intensity": hf["/results/cbox/pl/intensity"][:],
            }

            pl_spectra_dataset = xr.Dataset(
                {
                    "pl_spectrum": (
                        ["position", "wavelength"],
                        pl_spectra_data["intensity"],
                    )
                }
            )

            pl_spectra_dataset = pl_spectra_dataset.assign_coords(
                {"wavelength": pl_spectra_data["wavelength"]}
            )

            if process_params_coords:
                pl_spectra_dataset = pl_spectra_dataset.expand_dims(
                    "process_params", axis=0
                )
                pl_spectra_dataset = pl_spectra_dataset.assign_coords(
                    process_params_coords
                )

        if "results/cbox/uv_vis" in hf:
            uv_vis_data = {
                "wavelength": hf["/results/cbox/uv_vis/wavelength"][:],
                "absorption": hf["/results/cbox/uv_vis/absorption"][:],
            }

            uv_vis_dataset = xr.Dataset(
                {"absorption": (["position", "wavelength"], uv_vis_data["absorption"])}
            )

            uv_vis_dataset = uv_vis_dataset.assign_coords(
                {"wavelength": uv_vis_data["wavelength"]}
            )
            if process_params_coords:
                uv_vis_dataset = uv_vis_dataset.expand_dims("process_params", axis=0)
                uv_vis_dataset = uv_vis_dataset.assign_coords(process_params_coords)

        if "/results/xrd" in hf:
            xrd_data = {
                "2theta": hf["/results/xrd/2theta_Cu"][:],
                "intensity": hf["/results/xrd/int"][:],
            }

            xrd_dataset = xr.Dataset({"intensity": (["2theta"], xrd_data["intensity"])})
            xrd_dataset = xrd_dataset.assign_coords({"2theta": xrd_data["2theta"]})

            if process_params_coords:
                xrd_dataset = xrd_dataset.expand_dims("process_params", axis=0)
                xrd_dataset = xrd_dataset.assign_coords(process_params_coords)

    return {
        "cd_dataset": cd_dataset,
        "cbox_dataset": pl_img_dataset,
        "pl_spectra_dataset": pl_spectra_dataset,
        "uv_vis_dataset": uv_vis_dataset,
        "xrd_dataset": xrd_dataset,
    }


def build_dataTree(h5_files):
    """
    Builds an xarray.DataTree from a list of H5 files using coordinate inheritance.

    This function implements a "top-down" approach:
    1. It first concatenates all datasets, creating redundant coordinates.
    2. It then uses one of these concatenated datasets as a template to create a
       parent node containing only the shared coordinates.
    3. Finally, it creates child nodes by stripping the shared (now redundant)
       coordinates from each concatenated dataset.

    Args:
        h5_files (list): A list of paths to H5 files.

    Returns:
        xr.DataTree: A DataTree with shared coordinates in the root node.
    """
    # Asynchronously load the datasets from each H5 file.
    datasets_per_file = async_h5_processor_cache(build_datasets, use_cache=False)(
        h5_files
    )

    # Get all unique measurement keys (e.g., "xrd_dataset") from all files.
    all_keys = set(k for d in datasets_per_file for k in d.keys() if d)

    # Transpose the list of dictionaries into a dictionary of lists,
    # filtering out any missing (None) datasets.
    transposed_dict = {
        key: [d[key] for d in datasets_per_file if d and d.get(key) is not None]
        for key in all_keys
    }

    # Concatenate each list of datasets along a new 'process_params' dimension.
    # This creates datasets with redundant but complete coordinate information.
    concatenated_datasets = {
        k: xr.concat(v, dim="process_params") for k, v in transposed_dict.items() if v
    }

    if not concatenated_datasets:
        return xr.DataTree()  # Return an empty tree if no data was loaded.

    # Use the first available dataset as a template to define the parent node.
    template_dataset = next(iter(concatenated_datasets.values()))

    # Dynamically find the names of all coordinates associated with 'process_params'.
    process_params_keys = [
        k for k, v in template_dataset.coords.items() if v.dims == ("process_params",)
    ]

    # Create the parent node by keeping only the shared 'process_params' coordinates.
    parent_node = template_dataset.drop_dims(
        [dim for dim in template_dataset.dims if dim != "process_params"]
    )

    # Create the child nodes by stripping the now-redundant shared coordinates.
    child_nodes = {
        k: v.drop_vars(process_params_keys, errors="ignore")
        for k, v in concatenated_datasets.items()
    }

    # Assemble the final DataTree from the parent and child nodes.
    return xr.DataTree.from_dict({"/": parent_node, **child_nodes})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1:]
        print(build_dataTree(file_path))
    else:
        print("Please provide an H5 file path as the first argument")
