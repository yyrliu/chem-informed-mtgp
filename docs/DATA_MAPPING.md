# Data Mapping Documentation

This document outlines how the Large Parameter Space (LPS) preprocessing pipeline links various data sources (H5, CD, XRD, and Recipes) based on file metadata and naming conventions.

## 1. The Anchor: H5 Filenames

The H5 filename is the primary source of identity for a sample. The pipeline parses the filename to extract a unique **Sample ID**, which is then used to locate all other associated data.

### Parsing Logic
The pipeline expects filenames in the following format:
`{batch_name}_{sample_position}_{sample_id}_run{n}_spec_run.h5`

The filename is split by underscores `_`, and the components are assigned from right to left:
1.  **Run String:** The last segment (e.g., `run2`).
2.  **Sample ID:** The second-to-last segment. This is the critical key.
3.  **Sample Position:** The third-to-last segment.
4.  **Batch Name:** Everything preceding the sample position.

### Example Trace
**File:** `LPS-3\CBox\yrliu98_LPS-3_1_1_run2_spec_run.h5`

1.  **Remove Suffix:** `yrliu98_LPS-3_1_1_run2`
2.  **Split:** `['yrliu98', 'LPS-3', '1', '1', 'run2']`
3.  **Extraction:**
    *   `run_string` = `run2`
    *   **`sample_id`** = `1` (This is the key used for linking)
    *   `sample_position` = `1`
    *   `batch_name` = `yrliu98_LPS-3`

---

## 2. CD Data Mapping (String Matching)

Circular Dichroism (CD) files are located by searching the `CD` subdirectory for filenames that contain the **Sample ID** string.

*   **Search Logic:** The pipeline looks for files containing the `sample_id` string (e.g., `"1"`).
*   **Differentiation:** It requires two distinct files per sample, distinguished by the keywords `"front"` and `"back"`.
*   **Supported Formats:** `.txt` and `.jws` (JWS files are automatically converted to TXT).

**Example for Sample ID `1`:**
*   **Front File:** Matches `*1*front*.txt` (e.g., `sample_1_front.txt`)
*   **Back File:** Matches `*1*back*.txt` (e.g., `sample_1_back.txt`)

---

## 3. XRD Data Mapping (Spatial/Well Position)

X-Ray Diffraction (XRD) data can come from two sources:
- **GFRM files:** 2D detector images integrated using pyFAI with a PONI calibration file (primary source)
- **XY files:** Tabular XRD measurements (secondary source for comparison)

Both are mapped based on a physical well-plate layout logic derived from the filename.

### The Logic
XRD files contain a well position in their filename (e.g., `A01`, `C04`). The pipeline converts this position into a linear **Sample Index** to match the H5 `sample_id`.

**Formula:**
$$ \text{Sample ID} = (\text{Row Index} \times 4) + \text{Column Index} + 1 $$

*   **Rows (A-F):** A=0, B=1, C=2, D=3, E=4, F=5
*   **Columns (01-04):** 01=0, 02=1, 03=2, 04=3

### Data Storage Strategy
The pipeline prioritizes GFRM data when available:

**If GFRM file exists:**
- `results/xrd/` contains GFRM-integrated data (primary)
- `results/xrd_xy/` contains XY data if available (secondary, for validation/comparison)

**If GFRM file does NOT exist but XY file exists:**
- `results/xrd/` contains XY data (fallback)
- No `results/xrd_xy/` group is created

**If neither exists:**
- No XRD data is stored

Metadata (`file_info`) is stored as HDF5 attributes on each group rather than as separate datasets.

### Example Trace (Sample ID `1`)
To find the XRD files for Sample ID `1`:
1.  **Target ID:** `1`
2.  **Reverse Calculation:**
    *   $1 = (\text{Row} \times 4) + \text{Col} + 1$
    *   $0 = (\text{Row} \times 4) + \text{Col}$
    *   Result: **Row 0 (A)**, **Col 0 (01)**
3.  **Target Filenames:** 
    - **GFRM:** The pipeline looks for a file containing **`A01`** ending in `_000.gfrm` (e.g., `LPS_3_A01_000.gfrm`)
    - **XY:** The pipeline derives the XY filename by replacing `_000.gfrm` with `_merge.xy` (e.g., `LPS_3_A01_merge.xy`)

---

## 4. Recipe Parameters (Index Mapping)

Process parameters are loaded from a `recipes.json` file located in the base directory.

*   **Structure:** The JSON file contains a flat list of recipe objects.
*   **Logic:** The mapping is **positional (1-based index)**.
    *   The `sample_id` extracted from the H5 file is converted to an integer.
    *   This integer determines which item in the JSON list is used.

**Example for Sample ID `1`:**
*   **Target:** The **1st item** in the `recipes.json` list (Array Index 0).

---

## Summary Table: Example Trace

| Data Source | Source File / Logic | Key Identifier | Resulting Match |
| :--- | :--- | :--- | :--- |
| **H5 File** | `yrliu98_LPS-3_1_1_run2_spec_run.h5` | **Filename Parse** | **Sample ID: `1`** |
| **CD Data** | `CD/` folder search | String match `"1"` | `...1...front...txt` |
| **results/xrd/** | GFRM if exists, else XY | Well Calc `A01` | `...A01_000.gfrm` (if available) or `...A01_merge.xy` |
| **results/xrd_xy/** | XY data (only if GFRM also exists) | Well Calc `A01` | `...A01_merge.xy` |
| **Recipes** | `recipes.json` | List Index `0` | 1st Item in JSON list |
