import argparse
import json
import logging
import sys
import re
from pathlib import Path
from typing import Optional, Dict
from joblib import Parallel, delayed

# Add project root to sys.path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from scripts.preprocessing import process_single_h5_file_wrapper
except ImportError:
    # Try relative import if running as module
    from preprocessing import process_single_h5_file_wrapper


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def scan_cbox_h5_files(base_dir: Path) -> Dict[int, Path]:
    """
    Scan CBox directory for all H5 files and map sample_id to file path.
    Returns a dictionary {sample_id: file_path} of the highest run for each sample.
    """
    cbox_dir = base_dir / "CBox"
    mapping = {}

    # Store all runs: {sample_id: [(run_num, path), ...]}
    sample_runs = {}

    if not cbox_dir.exists():
        return mapping

    # Find all h5 files
    for h5_path in cbox_dir.rglob("*.h5"):
        try:
            # Logic from preprocessing.py
            clean_name = h5_path.name
            if not clean_name.endswith("_spec_run.h5"):
                continue

            parts = clean_name[:-12].split("_")
            if len(parts) < 3:
                continue

            sample_id_str = parts[-2]

            try:
                sample_id = int(sample_id_str)
                run_num = extract_run_number(clean_name)
                # If extraction fails (e.g. unexpected format), assume 0 or 1
                if run_num == -1:
                    run_num = 0

                if sample_id not in sample_runs:
                    sample_runs[sample_id] = []
                sample_runs[sample_id].append((run_num, h5_path))

            except ValueError:
                continue

        except Exception:
            continue

    # Process collected runs to find best run
    for sample_id, runs in sample_runs.items():
        # Sort desc by run number
        runs.sort(key=lambda x: x[0], reverse=True)

        # Pick the highest run regardless of validity
        best_run_num, best_path = runs[0]
        mapping[sample_id] = best_path

    return mapping


def extract_run_number(filename: str) -> int:
    """
    Extract run number from filename.
    Expects format ..._run(\d+)_spec_run.h5
    Returns -1 if not found.
    """
    match = re.search(r"_run(\d+)_spec_run\.h5$", filename)
    if match:
        return int(match.group(1))
    return -1


def find_h5_file(
    base_dir: Path,
    h5_filename: str,
    sample_id: Optional[int] = None,
    scan_map: Optional[Dict[int, Path]] = None,
) -> Optional[Path]:
    """
    Find H5 file:
    1. Try exact name match (nested) via glob (if valid).
    2. Try pattern with run number: XXX_run*_spec_run.h5 (highest run).
    3. Fallback: look up in scan_map via sample_id.
    """
    cbox_dir = base_dir / "CBox"
    if not cbox_dir.exists():
        return None

    h5_stem = Path(h5_filename).stem  # e.g. "Base_spec_run" or "Base"
    h5_name = Path(h5_filename).name

    # 1. Try exact filename match anywhere in CBox
    # Use glob to handle nesting
    matches = list(cbox_dir.glob(f"**/{h5_name}"))
    if matches:
        if len(matches) > 1:
            logging.warning(
                f"  Found multiple exact matches for {h5_name}, using first."
            )
        return matches[0]

    # 2. Try run number pattern
    # Construct base name "XXX" from "XXX_spec_run.h5"
    if h5_stem.endswith("_spec_run"):
        base_name = h5_stem[:-9]  # removes "_spec_run"
        pattern = f"**/{base_name}_run*_spec_run.h5"

        matches = list(cbox_dir.glob(pattern))
        if matches:
            # Sort by run number descending
            # We want the highest run number
            matches_with_run = []
            for m in matches:
                run_num = extract_run_number(m.name)
                # Filter out those where extract_run_number failed if we want strict matching
                if run_num != -1:
                    matches_with_run.append((run_num, m))

            if matches_with_run:
                # Sort by run number desc
                matches_with_run.sort(key=lambda x: x[0], reverse=True)

                # Select highest run
                best_run_num, best_match = matches_with_run[0]

                logging.info(f"  Found match via run pattern: {best_match.name}")
                return best_match

    # 3. Fallback using sample ID
    if sample_id is not None and scan_map is not None:
        if sample_id in scan_map:
            logging.info(
                f"  Fallback: Found file for sample {sample_id} via ID scan: {scan_map[sample_id].name}"
            )
            return scan_map[sample_id]

    return None


def process_mapping(
    mapping_file: str, poni_file: Optional[str] = None, n_jobs: int = -1
):
    """
    Read mapping file and process H5 files in parallel.
    """
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        logging.error(f"Mapping file not found: {mapping_path}")
        return

    base_dir = mapping_path.parent

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read mapping file: {e}")
        return

    tasks = []

    print(f"Loaded mapping with {len(mapping)} entries from {mapping_path.name}")
    print(f"Base directory: {base_dir}")

    # helper to find files by ID
    print("Scanning CBox for H5 files...")
    h5_scan_map = scan_cbox_h5_files(base_dir)
    print(f"Scanned {len(h5_scan_map)} H5 files with valid sample IDs.")

    # Sort keys to ensure deterministic order logic if needed, usually int keys in string format
    sorted_keys = sorted(mapping.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for sample_key in sorted_keys:
        sample_data = mapping[sample_key]
        files = sample_data.get("files", {})
        recipe = sample_data.get("recipe", {})

        # NOTE: Even if 'h5' is missing in mapping, if we have sample ID, we might find it via scan_map.
        # But user requested confirming mapped files exist.
        # Let's support discovering H5 if missing in mapping but ID is present?
        # The prompt implies "confirm mapped files exist", meaning they are in the mapping.

        if not files and sample_key.isdigit():
            # Try to recover if files dict is missing but we have sample ID
            files = {}
            # But we need at least h5_filename to start search?
            # scan_map uses sample ID.

        h5_path_str = files.get("h5")

        # If h5 string is present, use it for filename.
        # If NOT present, but we have sample ID, maybe we can find it in scan_map?

        sample_id_int = int(sample_key) if sample_key.isdigit() else None

        if not h5_path_str:
            if sample_id_int and sample_id_int in h5_scan_map:
                found_h5 = h5_scan_map[sample_id_int]
                logging.info(
                    f"  Found H5 for sample {sample_key} via scan (not in mapping): {found_h5.name}"
                )
            else:
                logging.warning(
                    f"No H5 file specified or found for sample {sample_key}"
                )
                continue
        else:
            h5_filename = Path(h5_path_str).name
            # 1.1 confirm that the mapped files exist, for h5 files, use glob
            found_h5 = find_h5_file(
                base_dir, h5_filename, sample_id=sample_id_int, scan_map=h5_scan_map
            )

        if not found_h5:
            logging.warning(
                f"✗ Could not locate H5 matching '{h5_path_str}' or sample ID {sample_key} in {base_dir}/CBox"
            )
            continue

        # Verify other files and build associated_files dict
        associated_files = {
            "cd_front": None,
            "cd_back": None,
            "xrd": None,
            "xrd_gfrm": None,
        }

        # Check CD Front
        if files.get("cd_front"):
            path_str = files["cd_front"]
            # If absolute, verify. If relative, join with base_dir (assumed, though usually paths in mapping are absolute based on sample)
            # The context shows absolute paths in mapping.
            p = Path(path_str)
            if not p.is_absolute():
                p = base_dir / p

            # Check for JWS extension and convert if necessary
            if p.suffix.lower() == ".jws":
                # Convert JWS to TXT using jws_converter module
                try:
                    from chem_informed_mtgp.jws_converter import convert_jws_to_txt

                    # Create txt path (e.g. file.txt)
                    txt_p = p.with_suffix(".txt")

                    if not txt_p.exists():
                        # Only convert if txt doesn't exist? Or always to be safe?
                        # Let's assume if it exists it's good, but maybe we should ensure it's up to date.
                        # For now, just convert if JWS exists.
                        if p.exists():
                            logging.info(f"  Converting JWS to TXT: {p.name}")
                            convert_jws_to_txt(str(p), str(txt_p))
                            p = txt_p  # Use the txt file
                        else:
                            logging.warning(f"  CD front file missing (JWS): {p}")
                    else:
                        p = txt_p  # Use existing txt

                except ImportError:
                    logging.warning(
                        "  Could not import jws_converter. CD data might fail if JWS format."
                    )
                except Exception as e:
                    logging.warning(f"  Failed to convert JWS: {e}")

            if p.exists():
                associated_files["cd_front"] = str(p)
            else:
                logging.warning(f"  CD front file missing: {p}")

        # Check CD Back
        if files.get("cd_back"):
            path_str = files["cd_back"]
            p = Path(path_str)
            if not p.is_absolute():
                p = base_dir / p

            # Check for JWS extension and convert if necessary
            if p.suffix.lower() == ".jws":
                # Convert JWS to TXT using jws_converter module
                try:
                    from chem_informed_mtgp.jws_converter import convert_jws_to_txt

                    txt_p = p.with_suffix(".txt")

                    if not txt_p.exists():
                        if p.exists():
                            logging.info(f"  Converting JWS to TXT: {p.name}")
                            convert_jws_to_txt(str(p), str(txt_p))
                            p = txt_p
                        else:
                            logging.warning(f"  CD back file missing (JWS): {p}")
                    else:
                        p = txt_p

                except ImportError:
                    pass
                except Exception as e:
                    logging.warning(f"  Failed to convert JWS: {e}")

            if p.exists():
                associated_files["cd_back"] = str(p)
            else:
                logging.warning(f"  CD back file missing: {p}")

        # Check XRD
        if files.get("xrd"):
            path_str = files["xrd"]
            p = Path(path_str)
            if not p.is_absolute():
                p = base_dir / p

            if p.exists():
                if p.suffix.lower() == ".gfrm":
                    associated_files["xrd_gfrm"] = str(p)
                elif p.suffix.lower() == ".xy":
                    associated_files["xrd"] = str(p)
                else:
                    # Fallback or assumption? process_single_h5_file typically handles .xy
                    associated_files["xrd"] = str(p)
            else:
                # Check if maybe the mapping has a path that doesn't exist but the file is there?
                # No, we trust the existence check.
                logging.warning(f"  XRD file missing: {p}")

        tasks.append(
            {
                "h5_file_path": str(found_h5),
                "base_dir": str(base_dir),
                "poni_file": poni_file,
                "associated_files": associated_files,
                "process_params": recipe,
            }
        )

    print(f"Found {len(tasks)} valid tasks ready for processing.")

    if not tasks:
        print("No tasks to process.")
        return

    print("Starting processing...")

    # 2. Process with joblib.Parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_h5_file_wrapper)(**task) for task in tasks
    )

    success_count = sum(1 for r in results if r[0])
    print(
        f"\nProcessing Summary: {success_count}/{len(tasks)} files processed successfully."
    )

    for success, msg in results:
        # Print logs (already formatted in wrapper)
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process H5 files based on a data mapping JSON."
    )
    parser.add_argument("mapping_file", help="Path to data_mapping.json")
    parser.add_argument(
        "--poni", help="Path to PONI file for GFRM processing", default=None
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: all cpus)",
    )

    args = parser.parse_args()

    setup_logging()
    process_mapping(args.mapping_file, args.poni, args.n_jobs)
