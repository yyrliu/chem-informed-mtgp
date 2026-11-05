from pathlib import Path
from typing import Optional

try:
    from jws2txt.helpers.helpers import JWSFile
except ImportError:
    print("Warning: 'jws2txt' package not found. JWS file conversion will be skipped.")
    print("Please install it with: pip install jws2txt")
    JWSFile = None


def find_and_convert_cd_file(cd_dir: Path, sample_id: str, side: str) -> Optional[str]:
    """
    Finds a CD file for a given sample ID and side (front/back).

    It first looks for a .txt file. If not found, it looks for a .jws file,
    converts it to .txt, and returns the path to the new .txt file.

    Args:
        cd_dir: The base directory to search for CD files.
        sample_id: The 13-character sample identifier.
        side: The measurement side, either 'front' or 'back'.

    Returns:
        The path to the .txt file as a string, or None if no suitable
        .txt or .jws file is found.
    """
    if not cd_dir.exists() or JWSFile is None:
        return None

    txt_pattern = f"{side}-{sample_id}.txt"
    jws_pattern = f"{side}-{sample_id}.jws"

    # Prioritize finding the .txt file first
    txt_files = list(cd_dir.rglob(txt_pattern))
    if txt_files:
        return str(txt_files[0])

    # If .txt not found, look for .jws and convert it
    jws_files = list(cd_dir.rglob(jws_pattern))
    if jws_files:
        jws_file = jws_files[0]
        output_txt_path = jws_file.with_suffix(".txt")
        convert_jws_to_txt(str(jws_file), str(output_txt_path))
        return str(output_txt_path)

    return None


def convert_jws_to_txt(input_path: str, output_path: str):
    """
    Directly converts a JWS file to TXT.

    Args:
        input_path: Path to the input .jws file
        output_path: Path to the output .txt file
    """
    if JWSFile is None:
        raise ImportError("'jws2txt' package not found or failed to load")

    JWSFile(input_path).write_data(output_path)
    print(f"  Converted {Path(input_path).name} to {Path(output_path).name}")
