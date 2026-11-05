#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CD Parser for Circular Dichroism spectroscopy data."""

import logging
from pathlib import Path

import xarray as xr
from natsort import natsorted
from crucible.parsers.base import BaseParser
import h5py

logger = logging.getLogger(__name__)


class CBoxParser(BaseParser):
    """Parser for Circular Dichroism (CD) spectroscopy data files.

    Supports parsing of CD data in .jws and .txt formats, with handling for
    both single measurements and front/back measurement pairs. Calculates
    g-factors and processes LD/LB (Linear Dichroism/Linear Birefringence) corrections.
    """

    # Define parser defaults
    _measurement = "autobot_cbox"
    _data_format = "h5"
    _instrument_name = "SpinBot"

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

        # Validate number of files and naming convention
        if len(self.files_to_upload) != 1:
            raise ValueError("Exactly one input file must be provided for CBoxParser")

        file_path = Path(self.files_to_upload[0])

        datasets = {}

        with h5py.File(file_path, "r") as hf:
            h5_mfid = hf.attrs.get("unique_id", None)
            if self.mfid and h5_mfid and self.mfid != h5_mfid:
                logger.warning(
                    f"MFID mismatch: provided mfid {self.mfid} does not match mfid in file {h5_mfid}, using provided mfid.",
                )
            elif h5_mfid:
                self.mfid = h5_mfid  # Use mfid from file if not provided
            else:
                raise ValueError(
                    "MFID not found in file attributes and no MFID provided to parser."
                )

            if "measurement/spec_run/wl_spectra" in hf:
                datasets["wl_data"] = xr.Dataset(
                    {
                        "wl_dark_int_time": hf["measurement/spec_run/wl_dark_int_time"][
                            ()
                        ],
                        "wl_dark_spectrum": hf["measurement/spec_run/wl_dark_spectrum"][
                            :
                        ],
                        "wl_samp_int_times": hf[
                            "measurement/spec_run/wl_spectra_int_times"
                        ][:],
                        "wl_samp_spectrum": hf["measurement/spec_run/wl_spectra"][:],
                        "wl_ref_int_time": hf["measurement/spec_run/wl_ref_int_time"][
                            ()
                        ],
                        "wl_ref_spectrum": hf["measurement/spec_run/wl_ref_spectrum"][
                            :
                        ],
                        "wl_wave_axis": hf["measurement/spec_run/wl_wls"][:],
                    }
                )

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
