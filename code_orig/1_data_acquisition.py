import re
import csv
from edgar import set_identity, Company

cik_list = []

# Read all ciks into the cik_list array
def read_cik():
    with open('../data/cik_all.csv', newline='\n', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            cik_list.append(row[0])


# Main program
if __name__ == "__main__":
    set_identity("<email>") # Required by the SEC

    read_cik()
    year = 2024 # Item1C section required from 2024

    item = open(f'../data/item1c_{year}_raw.csv', 'a')

    # Go throught the list of all cik's
    for cik in cik_list[:]:
        print(f"Processing {cik}")
        # 1. Get all company-related data to a given CIK
        company = Company(cik)

        # 2. Extract all 10-K filings for the company
        filings = company.get_filings(form="10-K")

        # 3. Find the 10-K report for a given year
        filing = next((f for f in filings if f.filing_date.year == year), None)

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
                item.write(f"{cik}|")
                item.write(f"{year}|")
                item.write(f"{fil_date}|")
                item.write(f"{rep_date}|")
                item.write(f"{item_text}\n")
            except:
                print(f"--- {cik}: No Item1C found ---")
                item.write(f"ERROR\n")

            item.flush()

    item.close()
