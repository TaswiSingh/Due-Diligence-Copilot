from pathlib import Path
from bs4 import BeautifulSoup


# --------------------------------------------------
# File paths
# --------------------------------------------------

input_file = Path("data/raw/apple_10k.html")
output_file = Path("data/processed/apple_10k.txt")


# --------------------------------------------------
# Check input file
# --------------------------------------------------

if not input_file.exists():
    raise FileNotFoundError(
        f"File not found: {input_file}"
    )


# --------------------------------------------------
# Read the HTML file
# --------------------------------------------------

html = input_file.read_text(
    encoding="utf-8"
)

print("HTML characters:", len(html))


# --------------------------------------------------
# Parse HTML using BeautifulSoup
# --------------------------------------------------

soup = BeautifulSoup(
    html,
    "html.parser"
)


# --------------------------------------------------
# Remove elements that don't contain useful text
# --------------------------------------------------

for element in soup(["script", "style", "noscript"]):
    element.decompose()


# --------------------------------------------------
# Extract text
# --------------------------------------------------

text = soup.get_text(
    separator="\n"
)


# --------------------------------------------------
# Clean the extracted text
# --------------------------------------------------

lines = []

for line in text.splitlines():

    line = line.strip()

    if line:
        lines.append(line)


clean_text = "\n".join(lines)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Save cleaned text
# --------------------------------------------------

output_file.write_text(
    clean_text,
    encoding="utf-8"
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("Clean text characters:", len(clean_text))
print("Saved to:", output_file)