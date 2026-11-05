#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CD Parser for Circular Dichroism spectroscopy data."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from natsort import natsorted
from crucible.parsers.base import BaseParser
from jws2txt.helpers.helpers import JWSFile


logger = logging.getLogger(__name__)


class CDParser(BaseParser):
    """Parser for Circular Dichroism (CD) spectroscopy data files.

    Supports parsing of CD data in .jws and .txt formats, with handling for
    both single measurements and front/back measurement pairs. Calculates
    g-factors and processes LD/LB (Linear Dichroism/Linear Birefringence) corrections.
    """

    # Define parser defaults
    _measurement = "cd_spectroscopy"
    _data_format = "jws"
    _instrument_name = None

    def parse(self):
        """
        Parse CD data files and extract metadata.

        This method is automatically called during initialization.
        All instance variables (self.mfid, self.files_to_upload, etc.)
        are already set and can be accessed.

        Raises:
            ValueError: If input files are invalid or in unexpected format.
        """
        # Validate input files
        if not self.files_to_upload:
            raise ValueError("No input files provided")

        files_to_upload_paths = [Path(f) for f in self.files_to_upload]
        if not all(f.suffix in [".txt", ".jws"] for f in files_to_upload_paths):
            raise ValueError("All input files must be .txt or .jws format")

        # Validate number of files and naming convention
        if len(files_to_upload_paths) == 1:
            pass  # Single measurement is valid
        elif len(files_to_upload_paths) == 2:
            # For paired measurements, validate front/back naming convention
            files_to_upload_paths.sort(key=lambda f: f.stem, reverse=True)
            if not (
                files_to_upload_paths[0].stem.startswith("front")
                and files_to_upload_paths[1].stem.startswith("back")
            ):
                raise ValueError(
                    "If two files are provided, they must be 'front' and 'back' measurements"
                )
        else:
            raise ValueError(
                "Too many input files provided. Only 'front' and 'back' measurements are expected."
            )

        # Convert .jws files to .txt format if needed
        txt_files = []
        for f in files_to_upload_paths:
            if f.suffix == ".jws":
                temp_cd_output = f.with_suffix(".txt")
                try:
                    JWSFile(str(f)).write_data(str(temp_cd_output), delimiter=",")
                    logger.info(f"Converted {f.name} to {temp_cd_output.name}")
                except FileExistsError:
                    logger.warning(
                        f"File '{temp_cd_output}' already exists. Skipping conversion of '{f}'."
                    )
                    temp_cd_output = f.with_suffix(".txt")
                txt_files.append(temp_cd_output)
            elif f.suffix == ".txt":
                txt_files.append(f)

        # Process single measurement file
        if len(txt_files) == 1:
            df = pd.read_csv(
                txt_files[0],
                sep="\s+",
                skiprows=3,
                names=["wavelength", "CD", "HT", "abs"],
            )
            ds = df.set_index("wavelength").to_xarray()
            ds = ds.expand_dims(measurement=[txt_files[0].stem])
            ds["g_factor"] = ds["CD"] / (ds["abs"] * 32980)

        # Process paired front/back measurement files
        elif len(txt_files) == 2:
            front_file = next((f for f in txt_files if f.stem.startswith("front")))
            back_file = next((f for f in txt_files if f.stem.startswith("back")))

            df_front = pd.read_csv(
                front_file,
                sep="\s+",
                skiprows=3,
                names=["wavelength", "CD", "HT", "abs"],
            )
            df_back = pd.read_csv(
                back_file,
                sep="\s+",
                skiprows=3,
                names=["wavelength", "CD", "HT", "abs"],
            )

            front = df_front.set_index("wavelength").to_xarray()
            back = df_back.set_index("wavelength").to_xarray()
            ds = xr.concat(
                [front, back], dim=pd.Index(["front", "back"], name="measurement")
            )

            # Create combined dataset with LD/LB correction
            # Average of front and back gives CD signal
            # Difference gives LD/LB (Linear Dichroism/Linear Birefringence) artifact
            ds_combined = xr.Dataset(
                {
                    "CD": (
                        ds.sel(measurement="front")["CD"]
                        + ds.sel(measurement="back")["CD"]
                    )
                    / 2,
                    "ldlb_CD": ds.sel(measurement="front")["CD"]
                    - ds.sel(measurement="back")["CD"],
                    "abs": (
                        ds.sel(measurement="front")["abs"]
                        + ds.sel(measurement="back")["abs"]
                    )
                    / 2,
                },
                coords={
                    "wavelength": ds["wavelength"].values,
                    "measurement": ["combined"],
                },
            )

            ds_combined["g_factor"] = ds_combined["CD"] / (ds_combined["abs"] * 32980)
            ds = xr.concat([ds, ds_combined], dim="measurement", data_vars="minimal")

        else:
            # Should never reach here due to earlier validation
            raise NotImplementedError("Unexpected number of files")

        # ds.to_zarr(txt_files[0].with_suffix('.zarr'), mode="w", consolidated=True)
        nc_file = txt_files[0].with_suffix(".nc")
        ds.to_netcdf(nc_file, mode="w")

        # Add domain-specific keywords (merges with user-provided keywords)
        self.add_keywords(["CD", "Circular Dichroism", "g-factor"])

        # Gather all files to upload (txt files + netcdf)
        self.files_to_upload = natsorted(
            set(self.files_to_upload + [str(f) for f in txt_files])
        )
        self.files_to_upload.insert(0, str(nc_file))

        # Generate thumbnail preview
        thumbnail_path = txt_files[0].with_suffix(".png").resolve()
        self._make_thumbnail(ds, thumbnail_path)
        self.thumbnail = str(thumbnail_path)

    @staticmethod
    def _make_thumbnail(ds: xr.Dataset, out_path: Path) -> None:
        """Generate a thumbnail preview plot of CD data.

        Creates a 3-panel plot showing CD signal, g-factor, and absorbance
        versus wavelength for all measurements in the dataset.

        Args:
            ds: xarray Dataset containing CD measurement data
            out_path: Path where the thumbnail PNG should be saved

        Returns:
            String path to the saved thumbnail file
        """
        import matplotlib.pyplot as plt

        # Create 3-panel plot: CD, g-factor, and absorbance
        fig, axes = plt.subplots(
            3, 1, figsize=(5, 6), sharex=True, layout="constrained"
        )
        ds["CD"].plot.line(x="wavelength", ax=axes[0])
        ds["g_factor"].plot.line(x="wavelength", ax=axes[1], add_legend=False)
        ds["abs"].plot.line(x="wavelength", ax=axes[2], add_legend=False)

        # Format top two panels
        for ax in axes[:2]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", direction="in", pad=-10)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

        axes[0].get_legend().set_title(None)

        # Final formatting and save
        fig.align_ylabels(axes)
        fig.get_layout_engine().set(h_pad=0.0, hspace=0.0)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Thumbnail saved to %s", out_path)
