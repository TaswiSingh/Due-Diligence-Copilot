from pathlib import Path
import json
import re


INPUT_FILE = Path("data/processed/apple_10k.txt")
OUTPUT_FILE = Path("data/processed/apple_chunks.json")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


TARGET_ITEMS = [
    "Item 1. Business",
    "Item 1A. Risk Factors",
    "Item 7. Managements Discussion and Analysis of Financial Condition and Results of Operations",
    "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8. Financial Statements and Supplementary Data",
]


def load_text():
    return INPUT_FILE.read_text(encoding="utf-8")


def normalize_heading(text):
    """
    Normalize whitespace so headings split across lines
    can still be compared.
    """
    return re.sub(r"\s+", " ", text).strip()


def find_item_positions(text):
    """
    Find occurrences of important 10-K headings.

    We search for headings that occur on their own line,
    which helps us avoid the Table of Contents.
    """

    lines = text.splitlines()

    positions = []

    for index, line in enumerate(lines):

        normalized = normalize_heading(line)

        for item in TARGET_ITEMS:

            if normalized.lower() == item.lower():

                positions.append({
                    "item": item,
                    "line": index
                })

    return lines, positions


def create_chunks(section_text):
    """
    Split section text into overlapping word-based chunks.
    """

    words = section_text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append(chunk_text)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def build_chunks(text):

    lines, positions = find_item_positions(text)

    print("Found section headings:")

    for position in positions:
        print(
            f"  {position['item']} "
            f"(line {position['line'] + 1})"
        )

    all_chunks = []

    chunk_id = 0

    for i, position in enumerate(positions):

        start_line = position["line"]

        if i + 1 < len(positions):
            end_line = positions[i + 1]["line"]
        else:
            end_line = len(lines)

        section_text = "\n".join(
            lines[start_line:end_line]
        ).strip()

        section_name = position["item"]

        chunks = create_chunks(section_text)

        for chunk_number, chunk in enumerate(chunks):

            chunk_id += 1

            all_chunks.append({
                "chunk_id": chunk_id,
                "company": "Apple Inc.",
                "filing_type": "10-K",
                "filing_year": 2025,
                "section": section_name,
                "section_chunk": chunk_number + 1,
                "text": chunk
            })

    return all_chunks


def main():

    print("Loading cleaned filing...")

    text = load_text()

    print("Characters:", len(text))

    print("Finding actual 10-K sections...")

    chunks = build_chunks(text)

    print()
    print("Total chunks:", len(chunks))

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()