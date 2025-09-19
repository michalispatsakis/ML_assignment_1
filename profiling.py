import os
import csv
from typing import List, Sequence


class CSVLogger:
    """
    Minimal CSV logger used across training scripts.

    - Writes headers once per file (if file is new or empty).
    - Appends rows thereafter.
    - Ensures parent directory exists.
    - Does not attempt cross-rank synchronization; callers should gate on rank 0.
    """

    def __init__(self, path: str, headers: List[str]):
        self.path = path
        self.headers = list(headers)
        # Ensure directory exists
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Write header if file does not exist or is empty
        if (not os.path.exists(self.path)) or os.path.getsize(self.path) == 0:
            with open(self.path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, row: Sequence):
        # Best-effort shape alignment: if lengths mismatch, truncate or pad with empty strings
        if len(row) != len(self.headers):
            if len(row) > len(self.headers):
                row = row[: len(self.headers)]
            else:
                row = list(row) + ["" for _ in range(len(self.headers) - len(row))]
        with open(self.path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
