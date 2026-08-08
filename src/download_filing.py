import requests
from pathlib import Path

# --------------------------------------------------
# 1. Apple's SEC CIK
# --------------------------------------------------

CIK = "0000320193"


# --------------------------------------------------
# 2. SEC submissions endpoint
# --------------------------------------------------

submissions_url = f"https://data.sec.gov/submissions/CIK{CIK}.json"


# --------------------------------------------------
# 3. Identify ourselves to SEC
# --------------------------------------------------

headers = {
    "User-Agent": "Due Diligence Copilot your_email@example.com"
}


# --------------------------------------------------
# 4. Get Apple's filing information
# --------------------------------------------------

response = requests.get(
    submissions_url,
    headers=headers
)

print("SEC status code:", response.status_code)

response.raise_for_status()

data = response.json()

print("Company:", data["name"])


# --------------------------------------------------
# 5. Find latest 10-K
# --------------------------------------------------

recent_filings = data["filings"]["recent"]

filing_date = None
accession_number = None
primary_document = None

for i, form in enumerate(recent_filings["form"]):

    if form == "10-K":

        filing_date = recent_filings["filingDate"][i]
        accession_number = recent_filings["accessionNumber"][i]
        primary_document = recent_filings["primaryDocument"][i]

        break


if primary_document is None:
    raise RuntimeError("No 10-K found")


print("\nLatest 10-K found!")
print("Filing date:", filing_date)
print("Accession number:", accession_number)
print("Primary document:", primary_document)


# --------------------------------------------------
# 6. Build SEC filing URL
# --------------------------------------------------

# Remove hyphens from accession number
accession_no_hyphens = accession_number.replace("-", "")

# Remove leading zeros from CIK
cik_without_zeros = str(int(CIK))

filing_url = (
    f"https://www.sec.gov/Archives/edgar/data/"
    f"{cik_without_zeros}/"
    f"{accession_no_hyphens}/"
    f"{primary_document}"
)

print("\nDownloading:")
print(filing_url)


# --------------------------------------------------
# 7. Download the actual 10-K
# --------------------------------------------------

filing_response = requests.get(
    filing_url,
    headers=headers
)

print("Filing download status:", filing_response.status_code)

filing_response.raise_for_status()


# --------------------------------------------------
# 8. Create output directory
# --------------------------------------------------

output_dir = Path("data/raw")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 9. Save the filing
# --------------------------------------------------

output_file = output_dir / "apple_10k.html"

output_file.write_text(
    filing_response.text,
    encoding="utf-8"
)

print("\n10-K downloaded successfully!")
print("Saved to:", output_file)
print("Characters:", len(filing_response.text))