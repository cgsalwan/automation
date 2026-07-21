# Record Lookup Filler

A small automation script that reconciles a source CSV export against a
target Excel workbook, filling in a blank lookup column wherever a match is
found — without touching rows that already have a value.

## Problem it solves

Many workflows produce two disconnected files:

- A **tracking workbook** (`.xlsx`) with a `Value` column that starts out
  blank for each record.
- A **raw system export** (`.csv`) — no header row, dozens of columns, and
  some rows still "in progress," carrying a placeholder instead of a real
  value.

Reconciling these by hand (VLOOKUP, copy-paste) is slow and error-prone,
especially since in-progress rows use a placeholder (`NA`) instead of a
real value. This script automates the lookup and prints a clear summary of
what happened.

## How it works

1. Reads the source CSV with **no header assumed** — Column A is the
   record ID, Column N is the value to pull across (configurable via
   constants at the top of the script if your export differs).
2. Skips any row where the value is the placeholder `"NA"` (still in
   progress), so those records are correctly reported as unmatched rather
   than filled with garbage.
3. Reads the target workbook **by header name** (`ID`, `Value` by default,
   also configurable), so it keeps working even if columns are reordered or
   new ones are added.
4. Only writes to **blank** cells in the value column — anything already
   filled is left alone and counted separately.
5. Saves the result to a new file, leaving the original workbook untouched.

## Usage

```bash
pip install -r requirements.txt

python record_lookup_filler.py --excel tracker.xlsx --csv export.csv
```

This saves the result as `tracker_updated.xlsx` in the same folder.

To control the output path explicitly:

```bash
python record_lookup_filler.py \
    --excel tracker.xlsx \
    --csv export.csv \
    --output tracker_filled.xlsx
```

### Example output

```
Loaded 298 ID -> Value mappings from CSV.
Matched & filled: 67
Already had a value (left untouched): 0
No match found in CSV: 234
Saved to tracker_updated.xlsx
```

## Requirements

- Python 3.10+
- [openpyxl](https://openpyxl.readthedocs.io/) (see `requirements.txt`)

## Notes / assumptions

- The target workbook's first sheet is used, and its first row must contain
  headers matching `ID_HEADER` and `VALUE_HEADER` (set at the top of the
  script — `"ID"` and `"Value"` by default).
- The source CSV's column layout (ID in column A, value in column N) is
  based on a specific export format. If your export differs, update the
  `CSV_ID_COL` / `CSV_VALUE_COL` constants near the top of the script.
- No data is modified in place — the script always writes to a new output
  file so you can diff or discard the result safely.

## License

MIT — feel free to adapt for your own workflow.
