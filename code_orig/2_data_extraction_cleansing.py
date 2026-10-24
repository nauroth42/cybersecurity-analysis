import re
import csv
from edgar import set_identity, Company

cik_list = []

def read_cik():
    with open('../data/cik_all.csv', newline='\n', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            cik_list.append(row[0])


def clean_text(raw_text):
    # 4. Get text and use a separator to keep words from sticking together
    # text = re.sub(r'[^\x00-\x7F]+', '', raw_text)
    text = raw_text.encode("ascii", "ignore").decode()

    # 5. Clean up whitespace and special characters
    # Replace non-breaking spaces (\xa0) with standard spaces
    text = text.replace('\xa0', ' ')

    # Replace delimiters (|) with (####)
    text = text.replace('|', '####')

    # Remove XBRL artifacts and redundant newlines
    text = re.sub(r'\n\s*\n', '\n', text)

    # Remove extra spaces
    text = re.sub(r' +', ' ', text)

    # [\r\n\t] matches carriage returns, newlines, and tabs
    # You can add other hex codes if needed, e.g., \x00-\x1f
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # 6. Optional: Remove page numbers or "Table of Contents" markers
    # This is tricky as patterns vary, but a basic regex can help:
    text = re.sub(r'Page \d+', '', text)

    return text[:200000]


# Main program
if __name__ == "__main__":
    set_identity("<email>") # Required by the SEC

    read_cik()
    year = 2024 # Item1C section required from 2024

    item = open(f'../data/item1c_{year}_raw.csv', 'r')
    item = open(f'../data/item1c_{year}_cln.csv', 'a')

    with open(f'../data/item1c_{year}_raw.csv', mode='r', ) as file:
        # Define the custom pipe delimiter
        csv_reader = csv.reader(file, delimiter='|')

        #No header row
        for row in csv_reader:
                    file.read(f"{cik}|")
                    item.write(f"{year}|")
                    item.write(f"{fil_date}|")
                    item.write(f"{rep_date}|")
                    item.write(f"{clean_text(item_text)}\n")
                except:
                    print(f"--- {cik} ITEM x ERROR ---")
                    item.write(f"ERROR\n")

                item.flush()

    item.close()
