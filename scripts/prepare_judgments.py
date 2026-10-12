"""Extract searchable text and basic metadata from downloaded judgments.

Run:
    python scripts/prepare_judgments.py
"""
import argparse
import json
import os
import re
from datetime import date

import pymupdf
from tqdm import tqdm


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(HERE, "data", "processed", "extracted")
DEFAULT_OUTPUT = os.path.join(HERE, "data", "processed", "judgments.jsonl")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def metadata_for(path):
    relative = os.path.relpath(path, DEFAULT_INPUT)
    parts = relative.split(os.sep)
    court = parts[0]
    source_group = parts[1] if len(parts) > 1 else "unknown"
    source_bench = parts[1] if court == "high_court" and len(parts) > 1 else None
    source_year = parts[1] if court == "supreme_court" and len(parts) > 1 else None
    match = DATE_PATTERN.search(os.path.basename(path))
    judgment_date = match.group(1) if match else None

    return {
        "judgment_id": os.path.splitext(os.path.basename(path))[0],
        "court": court,
        "source_group": source_group,
        "source_year": source_year,
        "source_bench": source_bench,
        "judgment_date": judgment_date,
        "source_file": relative.replace(os.sep, "/"),
    }


def extract_document(path):
    with pymupdf.open(path) as document:
        pages = [page.get_text("text") for page in document]
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Extracted PDF directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    args = parser.parse_args()

    pdfs = []
    for root, _, files in os.walk(args.input):
        pdfs.extend(
            os.path.join(root, name)
            for name in files
            if name.lower().endswith(".pdf")
        )
    pdfs.sort()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    processed = 0
    empty = 0
    errors = 0

    with open(args.output, "w", encoding="utf-8") as output:
        for path in tqdm(pdfs, desc="Extracting judgments"):
            try:
                text, page_count = extract_document(path)
                if not text:
                    empty += 1
                    continue
                record = metadata_for(path)
                record.update({
                    "page_count": page_count,
                    "character_count": len(text),
                    "text": text,
                })
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as error:
                errors += 1
                print(f"[error] {path}: {error}")

    print(f"Processed: {processed}")
    print(f"Empty PDFs skipped: {empty}")
    print(f"Errors: {errors}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
