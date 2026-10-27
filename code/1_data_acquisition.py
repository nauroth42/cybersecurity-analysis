"""
1_data_aquisition.py
=====================

SEC EDGAR 10-K "Item 1C" (Cybersecurity) Data Acquisition Script.

This script iterates over a list of company CIK (Central Index Key) numbers,
retrieves each company's 10-K filing for a target fiscal year from the SEC
EDGAR system (via the `edgar` Python package), and extracts the text of
"Item 1C" (Cybersecurity) from that filing.

Background
----------
Starting with fiscal year 2023 annual reports (i.e., 10-K filings made in
2024), the SEC requires public companies to disclose cybersecurity risk
management, strategy, and governance information under a new "Item 1C"
section. This script harvests that section's raw text across many
companies for downstream analysis (e.g., NLP, text mining, compliance
research).

Output
------
Results are appended to a pipe-delimited (`|`) CSV-like file named
``../data/item1c_<year>_raw.csv`` with one row per company, containing:

    CIK | Year | Filing Date | Report Date | Item 1C Text

If no Item 1C section can be found (or any error occurs while parsing),
the literal string ``ERROR`` is written to that company's row instead.

Dependencies
------------
- `edgar` (the `edgartools` package): provides `set_identity`, `Company`,
  and filing/object parsing utilities for SEC EDGAR data.
- Python standard library: `re`, `csv`.

Input Files
------------
- ``../data/cik_all.csv``: a single-column CSV file (UTF-8 with BOM) where
  each row contains one company CIK number to process.

Usage
-----
Run directly as a script::

    python 1_data_aquisition.py

Note: The SEC requires all automated EDGAR requests to identify the
requester (name and/or email) via `set_identity()`. Replace the
placeholder `"<email>"` with a valid contact email before running.
"""

# pip install edgartools

import re
import csv
from edgar import set_identity, Company

# Target fiscal year: Item 1C (Cybersecurity) disclosures are only
# required starting with 10-K filings made in 2024.
YEAR = 2024 # Item1C section required from 2024

# Global list that will hold every CIK (Central Index Key) read from the
# input CSV file. Populated by `read_cik()` and consumed by the main loop.
cik_list = []


def read_cik():
    """
    Read all company CIK numbers from the input CSV file into `cik_list`.

    Opens ``../data/cik_all.csv`` (encoded as UTF-8 with a byte-order mark,
    hence ``utf-8-sig``) and reads it as a comma-delimited CSV file. Each
    row is expected to contain a single CIK value in its first column
    (``row[0]``); that value is appended to the module-level `cik_list`.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return a value. It mutates the global
        `cik_list` in place by appending one entry per row read from the
        CSV file.

    Side Effects
    ------------
    - Reads from disk: ``../data/cik_all.csv``.
    - Appends to the module-level global variable `cik_list`.

    Raises
    ------
    FileNotFoundError
        If ``../data/cik_all.csv`` does not exist at the expected relative
        path.
    IndexError
        If any row in the CSV file is empty (i.e., has no columns), since
        `row[0]` would be out of range.

    Notes
    -----
    - The CSV reader is configured with `delimiter=','` and
      `quotechar='|'`. Since the pipe character (`|`) is not typically
      used for quoting in this file, this setting is effectively a no-op
      unless the source file uses `|` for quoting.
    - No header row is skipped; if the CSV file has a header row, it will
      be read as if it were a CIK value.
    """
    with open('../data/cik_all.csv', mode='r', newline='\n', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',', quotechar='|')

        for row in csv_reader:
            cik_list.append(row['cik'])


# ---------------------------------------------------------------------------
# Main program entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Main execution block.

    Workflow
    --------
    1. Identify this script to the SEC EDGAR system via `set_identity()`,
       as required by SEC fair-access rules for automated requests.
    2. Load the full list of CIKs to process via `read_cik()`.
    3. Open (in append mode) the output file for the target `year`, where
       extracted Item 1C text will be written.
    4. For each CIK in `cik_list`:
        a. Instantiate a `Company` object for that CIK.
        b. Retrieve all of the company's 10-K filings.
        c. Select the specific 10-K filing whose `filing_date.year`
           matches the target `year`.
        d. If such a filing exists, parse it into a structured `TenK`
           filing object (`filing.obj()`) and capture its filing date
           and report date.
        e. Attempt to extract the "Item 1C" (Cybersecurity) section text
           from the parsed filing object.
        f. Write a pipe-delimited record to the output file:
           ``CIK|Year|FilingDate|ReportDate|Item1CText``.
           On failure (e.g., missing section, parsing error), write
           ``ERROR`` instead.
    5. Flush the output file after each company to persist progress
       incrementally, then close the file once all CIKs are processed.

    Output File Format
    -------------------
    Each successful line in ``../data/item1c_<year>_raw.csv`` has the
    form (fields separated by ``|``, terminated by a newline)::

        <CIK>|<YEAR>|<FILING_DATE>|<REPORT_DATE>|<ITEM_1C_TEXT>

    Each failed/error line has the form::

        ERROR

    Note: because the error branch does not also write the CIK, year,
    or dates, error rows are NOT directly traceable back to a specific
    company from the output file alone; this may be worth revisiting for
    downstream analysis.

    Variables
    ---------
    year : int
        The target fiscal year for which 10-K filings and Item 1C
        sections are extracted. Hardcoded to 2024, since Item 1C
        disclosures were first required starting with 10-K filings made
        in that year.
    item : file object
        Open file handle (append mode, text) to the output CSV-like file
        ``../data/item1c_<year>_raw.csv``.
    cik : str
        The current CIK number being processed, drawn from `cik_list`.
    company : edgar.Company
        EDGAR company object corresponding to the current `cik`, used to
        query that company's filings.
    filings : edgar.EntityFilings (iterable)
        Collection of all 10-K filings associated with `company`.
    filing : edgar.Filing or None
        The specific 10-K filing whose filing year matches `year`, or
        `None` if no such filing exists for this company.
    tenk : edgar.TenK
        Structured, parsed representation of `filing`, providing
        dictionary-like access to individual report sections/items
        (e.g., `tenk['Item 1C']`).
    fil_date : datetime.date
        The SEC filing date of the located 10-K (`filing.filing_date`).
    rep_date : datetime.date
        The fiscal period/report date covered by the located 10-K
        (`filing.report_date`).
    item_text : str
        The raw extracted text of the "Item 1C" (Cybersecurity) section
        from the filing.

    Error Handling
    --------------
    - If no 10-K filing exists for the given `year`, `filing` remains
      `None`. Note: the subsequent block that reads `fil_date`/`rep_date`
      and attempts extraction is only entered when `filing` is truthy
      (see the `if filing:` guard), so a missing filing is silently
      skipped without writing any output line for that CIK.
    - If a filing exists but `tenk['Item 1C']` raises any exception
      (e.g., `KeyError` if the section is absent, or a parsing error),
      the `except` block catches it (bare `except`, catching all
      exception types), prints a diagnostic message to stdout, and
      writes the literal string ``ERROR\\n`` to the output file for that
      company.

    Side Effects
    ------------
    - Network I/O: queries the live SEC EDGAR system for each CIK's
      filings (rate-limited/politeness governed by `set_identity`).
    - Disk I/O: appends to ``../data/item1c_{year}_raw.csv`` after every
      processed company (via `item.flush()`), and closes the file at the
      end of the run.
    - Console output: prints progress and status messages for each CIK
      processed.

    Caveats / Potential Improvements
    ---------------------------------
    - The output file is opened in append (`'a'`) mode without first
      checking if it already exists; re-running the script will append
      duplicate data rather than overwrite it.
    - The bare `except:` clause suppresses all exception types
      indiscriminately, which can mask unexpected bugs (e.g., network
      errors) as if they were simply "Item 1C not found".
    - `set_identity("<email>")` must be updated with a real, valid
      contact email address before running, per SEC EDGAR's automated
      access policy.
    """

    # Identify this script/user to SEC EDGAR (required for automated access).
    # Replace "<email>" with an actual contact email before running.
    set_identity('mnaurot@siue.edu')

    # Populate the global `cik_list` with all CIKs to process.
    read_cik()

    # Open the output file in append mode so progress is preserved across
    # multiple runs (though see caveat above regarding duplicate rows).
    with open(f'../data/item1c_{YEAR}_raw.csv', 'w') as file:

        # Iterate over a shallow copy of the CIK list ([:]) to process each
        # company in turn.
        for cik in cik_list[:]:
            print(f"Processing {cik}")

            # 1. Get all company-related data to a given CIK
            company = Company(cik)

            # 2. Extract all 10-K filings for the company
            filings = company.get_filings(form="10-K")

            # 3. Find the 10-K report for a given year
            #    (first filing whose filing_date falls within `year`, or None)
            filing = next((f for f in filings if f.filing_date.year == YEAR), None)

            # 4. Parse the specific filing into a TenK object
            if filing:
                tenk = filing.obj()
                fil_date = filing.filing_date
                rep_date = filing.report_date

                # 5. Extract the item section
                # The library automatically parses the document structure!
                try:
                    item_text = tenk['Item 1C']

                    print(f"--- {cik}: Item1C extracted ---")
                    file.write(f"{cik}|")
                    file.write(f"{YEAR}|")
                    file.write(f"{fil_date}|")
                    file.write(f"{rep_date}|")
                    file.write(f"{[item_text]}\n")
                except Exception:
                    # Catches any failure extracting Item 1C (e.g., section
                    # missing from this filing, or a parsing error) and
                    # records a simple error marker instead of crashing the
                    # whole batch run.
                    print(f"--- {cik}: No Item1C found ---")

                # Flush after every company so partial progress is saved to
                # disk even if the script is interrupted mid-run.
                file.flush()