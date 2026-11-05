import json
import argparse
import re
from pathlib import Path


def get_well_position(sample_id, rows=None, cols=4):
    if rows is None:
        rows = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # robust integer conversion
    try:
        idx = int(sample_id) - 1
    except ValueError:
        return None

    if idx < 0:
        return None

    row_idx = idx // cols
    col_idx = idx % cols

    if row_idx >= len(rows):
        return None

    well = f"{rows[row_idx]}{col_idx + 1:02d}"
    return well


def find_file(directory, pattern_func, sample_id):
    """
    Finds a file in directory matching the pattern logic.
    pattern_func(sample_id) returns a regex. The result is a path relative to
    the batch directory (e.g. "CD/front-1.jws"), or "" if nothing matches.
    """
    if not directory.exists():
        return ""

    # We'll use glob for simple patterns, or listdir + regex for complex ones
    # Let's try to be flexible. pattern_func returns a regex string.
    regex = pattern_func(sample_id)
    compiled = re.compile(regex, re.IGNORECASE)

    matches = []
    for f in directory.iterdir():
        if f.is_file() and compiled.match(f.name):
            matches.append((Path(directory.name) / f.name).as_posix())

    if not matches:
        return ""

    # Return the file (if multiple, maybe warn or return first? H5 usually has one spec run)
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate data mapping from recipes JSON."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory of the experiment batch (e.g., LPS-5)",
    )
    parser.add_argument(
        "--recipes",
        type=str,
        default=None,
        help="Name of the recipes JSON file inside base_dir, or absolute path. If not specified, finds *recipe*.json.",
    )
    parser.add_argument(
        "--output", type=str, default="data_mapping.json", help="Output JSON filename."
    )
    parser.add_argument(
        "-n",
        "--no-verify",
        action="store_true",
        help="If set, generate mapping even if files are missing, using predicted filenames.",
    )

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()

    # Locate recipes file
    if args.recipes:
        recipes_path = Path(args.recipes)
        if not recipes_path.is_absolute():
            recipes_path = base_dir / recipes_path

        if not recipes_path.exists():
            print(f"Error: Recipes file not found at {recipes_path}")
            return
    else:
        # Search for *recipe*.json
        candidates = list(base_dir.glob("*recipe*.json"))
        if not candidates:
            print(f"Error: No *recipe*.json files found in {base_dir}")
            return

        if len(candidates) == 1:
            recipes_path = candidates[0]
        else:
            # Heuristic: Prefer "recipes.json"
            exact = base_dir / "recipes.json"
            if exact in candidates:
                recipes_path = exact
            else:
                print(
                    f"Error: Multiple recipe files found: {[c.name for c in candidates]}. Please specify one."
                )
                return

    print(f"Reading recipes from {recipes_path}")
    with open(recipes_path, "r") as f:
        recipes = json.load(f)

    mapping = {}

    # Directories
    cbox_dir = base_dir / "CBox"
    cd_dir = base_dir / "CD"
    xrd_dir = base_dir / "XRD"

    print(f"Scanning directories in {base_dir}...")

    for i, recipe in enumerate(recipes):
        sample_id = i + 1
        sid_str = str(sample_id)

        # 1. H5 (CBox) Mapping
        # Pattern: {batch}_{pos}_{id}_run{n}_spec_run.h5
        # Regex: ends with _{id}_run\d+_spec_run.h5 or similar.
        # To be safe: .*_{id}_run.*\.h5
        def h5_regex(sid):
            return rf".*_{sid}_run.*\.h5$"

        h5_file = find_file(cbox_dir, h5_regex, sid_str)

        if not h5_file and args.no_verify:
            # Construct a placeholder path
            h5_file = f"CBox/{base_dir.name}_sample_{sample_id}_spec_run.h5"

        # 2. CD Mapping
        # Front: .*front-{id}\.txt (or .jws)
        # Back: .*back-{id}\.txt (or .jws)
        # Note: listing showed names like 'front-9.txt'
        def cd_front_regex(sid):
            return rf".*front-{sid}\.(txt|jws)$"

        def cd_back_regex(sid):
            return rf".*back-{sid}\.(txt|jws)$"

        cd_front = find_file(cd_dir, cd_front_regex, sid_str)
        cd_back = find_file(cd_dir, cd_back_regex, sid_str)

        if args.no_verify:
            if not cd_front:
                cd_front = f"CD/front-{sid_str}.jws"
            if not cd_back:
                cd_back = f"CD/back-{sid_str}.jws"

        # 3. XRD Mapping
        # Based on Well Position
        well = get_well_position(sample_id)
        xrd_file = ""  # default to empty string if not found
        if well:
            # Pattern: .*_{well}_.*\.gfrm (or .xy)
            # Example: LPS_5_A01_000.gfrm
            def xrd_regex(w):
                return rf".*_{w}_.*\.gfrm$"

            def xrd_regex_xy(w):
                return rf".*_{w}_.*\.xy$"

            if xrd_dir.exists():
                xrd_file = find_file(xrd_dir, xrd_regex, well)

                # Fallback to .xy? listing showed .xy too. But .gfrm is primary per DATA_MAPPING.md
                if not xrd_file:
                    xrd_file = find_file(xrd_dir, xrd_regex_xy, well)

            if not xrd_file and args.no_verify:
                xrd_file = f"XRD/{base_dir.name}_{well}_000.gfrm"

        entry = {
            "sample_id": sample_id,
            "recipe": recipe,
            "files": {
                "h5": h5_file,
                "cd_front": cd_front,
                "cd_back": cd_back,
                "xrd": xrd_file,
            },
            "derived_metadata": {"well_position": well},
        }

        mapping[sample_id] = entry

    output_path = args.output
    if not Path(output_path).is_absolute():
        output_path = base_dir / output_path

    print(f"Writing mapping to {output_path}")
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
