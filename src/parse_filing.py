from pathlib import Path
from bs4 import BeautifulSoup
import re


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
# Read HTML
# --------------------------------------------------

html = input_file.read_text(
    encoding="utf-8"
)

# Fix common encoding artifacts
html = html.replace("â€™", "’")
html = html.replace("â€œ", "“")
html = html.replace("â€", "”")
html = html.replace("â€”", "—")
html = html.replace("â€“", "–")
html = html.replace("â˜’", "☒")
html = html.replace("â˜", "☐")

print("HTML characters:", len(html))


# --------------------------------------------------
# Parse HTML
# --------------------------------------------------

soup = BeautifulSoup(
    html,
    "html.parser"
)


# --------------------------------------------------
# Remove unnecessary HTML/XBRL elements
# --------------------------------------------------

tags_to_remove = [
    "script",
    "style",
    "noscript",
    "meta",
    "link",
    "title",
]

for tag in soup.find_all(tags_to_remove):
    tag.decompose()


# --------------------------------------------------
# Remove Inline XBRL metadata/header
# --------------------------------------------------

for tag in soup.find_all():
    tag_name = tag.name.lower() if tag.name else ""

    if tag_name in [
        "ix:header",
        "ix:hidden",
    ]:
        tag.decompose()


# --------------------------------------------------
# Remove hidden elements
# --------------------------------------------------

for tag in soup.find_all():

    if tag.has_attr("hidden"):
        tag.decompose()
        continue

    style = tag.get("style", "")

    if "display:none" in style.replace(" ", "").lower():
        tag.decompose()
        continue

    if "visibility:hidden" in style.replace(" ", "").lower():
        tag.decompose()


# --------------------------------------------------
# Extract visible text
# --------------------------------------------------

text = soup.get_text(
    separator="\n"
)


# --------------------------------------------------
# Clean whitespace
# --------------------------------------------------

lines = []

for line in text.splitlines():

    line = line.strip()

    if not line:
        continue

    # Remove excessive whitespace
    line = re.sub(
        r"\s+",
        " ",
        line
    )

    lines.append(line)


clean_text = "\n".join(lines)


# --------------------------------------------------
# Save cleaned text
# --------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Fix common mojibake encoding issues
clean_text = clean_text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")

output_file.write_text(
    clean_text,
    encoding="utf-8"
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "Clean text characters:",
    len(clean_text)
)

print(
    "Saved to:",
    output_file
)