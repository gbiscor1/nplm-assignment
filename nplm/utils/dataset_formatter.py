# nplm/utils/dataset_formatter.py
"""
Provides reusable utilities for formatting and saving standardized dataset records in JSONL.
"""
import json
from pathlib import Path
from typing import Iterable


def write_jsonl(file_path: Path, records: Iterable[dict]) -> Path:
    """
    write_jsonl: writes dataset records to a JSONL file using UTF-8 encoding.

    Params:
        file_path (Path): destination path for the JSONL file.
        records (Iterable[dict]): dataset records to serialize and write one record per line.

    Returns:
        Path (Path): path to the written JSONL file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return file_path