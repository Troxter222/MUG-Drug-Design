"""
Transformer Molecule Dataset Preparation Script
================================================

This script processes large molecular datasets containing SMILES strings and
converts them to the SELFIES representation suitable for training a
Transformer model.  It is designed to handle tens of millions of molecules
without loading the entire dataset into memory.  The pipeline streams
through input files, validates and filters molecules by token length,
collects a SELFIES vocabulary, and writes valid examples to a CSV file.

Key Features
------------

- **Streaming processing:** The script reads input files line by line and
  writes output immediately, avoiding loading entire datasets into RAM.
- **Flexible input formats:** Handles plain text, CSV, and TSV files.
  SMILES strings may appear either as the first column or in a column
  named ``smiles`` (case‐insensitive).  Headers are detected
  automatically.
- **SELFIES conversion and filtering:** Each SMILES string is converted
  into SELFIES; molecules are filtered by token length (configurable
  minimum and maximum), and invalid SMILES are skipped.
- **Vocabulary collection:** Builds a token vocabulary on the fly while
  streaming through the data.  Adds special tokens ``<pad>``, ``<sos>``,
  and ``<eos>`` used by many Transformer models.
- **Progress logging:** Emits progress messages every configurable
  number of lines (default: 1,000,000) and displays a tqdm progress bar.
- **Statistics:** Reports counts of total lines processed, valid
  molecules kept, and molecules skipped (too short, too long, or
  failed conversion).

Usage Example
-------------

To run the script on multiple input files and write the results to
``transformer_train.csv`` and ``vocab_transformer.json``:

.. code-block:: bash

   python transformer_data_pipeline.py \
       --input data1.txt data2.csv data3.tsv \
       --output_csv transformer_train.csv \
       --vocab_json vocab_transformer.json \
       --min_tokens 5 --max_tokens 150 --progress_interval 1000000

Requirements
------------

- Python ≥ 3.10
- Libraries: ``selfies``, ``tqdm``, ``json``, ``argparse``, ``pathlib``

The script avoids using pandas or other heavy libraries and is suitable
for production data pipelines.

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import selfies
from tqdm import tqdm


class Config:
    """Configuration parameters for the dataset preparation pipeline.

    Attributes
    ----------
    input_files : List[Path]
        List of paths to input files to process.  Files may be plain text,
        comma-separated values (CSV), or tab-separated values (TSV).
    output_csv : Path
        Path to the output CSV file where valid SELFIES sequences will be
        written.  The CSV will have a single column named ``selfies``.
    vocab_json : Path
        Path to the JSON file where the token vocabulary will be saved.
        The JSON will contain a mapping from tokens to integer indices
        (``token_to_id``) and the reverse mapping (``id_to_token``).
    min_tokens : int
        Minimum number of tokens required for a SELFIES sequence to be kept.
    max_tokens : int
        Maximum number of tokens allowed for a SELFIES sequence to be kept.
    progress_interval : int
        Number of lines to process before logging progress and updating the
        tqdm progress bar.
    """

    def __init__(
        self,
        input_files: List[Path],
        output_csv: Path,
        vocab_json: Path,
        min_tokens: int = 5,
        max_tokens: int = 150,
        progress_interval: int = 1_000_000,
    ) -> None:
        self.input_files = input_files
        self.output_csv = output_csv
        self.vocab_json = vocab_json
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.progress_interval = progress_interval


class Statistics:
    """Tracks processing statistics for reporting purposes."""

    def __init__(self) -> None:
        self.total: int = 0
        self.kept: int = 0
        self.failed: int = 0
        self.too_short: int = 0
        self.too_long: int = 0

    def report(self) -> str:
        """Return a formatted string summarizing the collected statistics."""
        return (
            f"Processed: {self.total:,} | "
            f"Kept: {self.kept:,} | "
            f"Failed: {self.failed:,} | "
            f"Too short: {self.too_short:,} | "
            f"Too long: {self.too_long:,}"
        )


def detect_delimiter(path: Path) -> Optional[str]:
    """Heuristically determine the delimiter for a CSV/TSV file.

    For files ending with `.csv`, ``','`` is returned.  For `.tsv`, ``'\t'``
    is returned.  Otherwise ``None`` is returned, signalling that the
    caller should treat the file as whitespace-delimited text.

    Parameters
    ----------
    path : Path
        The file whose delimiter is to be detected.

    Returns
    -------
    Optional[str]
        The delimiter character if known, otherwise ``None``.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix == ".tsv":
        return "\t"
    return None


def iter_smiles_from_delimited(
    file_path: Path,
    delimiter: str,
    stats: Statistics,
) -> Iterable[str]:
    """Yield SMILES strings from a delimited file (CSV or TSV).

    This generator reads the file row by row using the built-in
    ``csv`` module.  It detects whether the first row is a header
    containing a ``smiles`` column and selects the appropriate column
    accordingly.  If no header is detected, the first column is used.

    Parameters
    ----------
    file_path : Path
        Path to the file to read.
    delimiter : str
        The field delimiter (`,` for CSV or ``\t`` for TSV).
    stats : Statistics
        A ``Statistics`` instance used to update the count of total lines
        processed.  The total will be incremented for each row read from
        the file.

    Yields
    ------
    str
        The SMILES string extracted from the current row.
    """
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first_row = next(reader)
        except StopIteration:
            return  # empty file

        # Determine if the first row is a header by looking for a "smiles" column
        header_lower = [cell.strip().lower() for cell in first_row]
        if "smiles" in header_lower:
            smiles_index = header_lower.index("smiles")
            # skip header row
        else:
            smiles_index = 0
            # treat the first row as data
            row = first_row
            # update stats and yield
            stats.total += 1
            if len(row) > smiles_index:
                yield row[smiles_index]

        # Process remaining rows
        for row in reader:
            stats.total += 1
            if not row:
                continue
            if len(row) <= smiles_index:
                continue
            yield row[smiles_index]


def iter_smiles_from_text(
    file_path: Path,
    stats: Statistics,
) -> Iterable[str]:
    """Yield SMILES strings from a whitespace-separated text file.

    Each line is split on whitespace; the first token is assumed to be the
    SMILES string.  Empty lines are skipped.  The total line count in
    ``stats`` is updated for each non-empty line.

    Parameters
    ----------
    file_path : Path
        Path to the text file.
    stats : Statistics
        Statistics instance to update the total count.

    Yields
    ------
    str
        The SMILES string extracted from the current line.
    """
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            stats.total += 1
            # split on any whitespace; take first token as SMILES
            parts = stripped.split()
            if parts:
                yield parts[0]


def process_smiles_stream(
    smiles_iter: Iterable[str],
    writer: csv.DictWriter,
    vocab: Dict[str, int],
    config: Config,
    stats: Statistics,
    pbar: tqdm,
    progress_counter: Dict[str, int],
) -> None:
    """Process a stream of SMILES strings, converting to SELFIES and writing to CSV.

    This function iterates over an iterable of SMILES strings, converts each
    to SELFIES using the ``selfies`` library, splits the SELFIES into
    tokens, filters by token count, writes valid sequences to the CSV
    output, and updates the vocabulary.  It also updates statistics and
    periodically logs progress via the provided tqdm progress bar.

    Parameters
    ----------
    smiles_iter : Iterable[str]
        An iterable yielding SMILES strings.
    writer : csv.DictWriter
        CSV writer configured to write rows with a single ``selfies`` field.
    vocab : Dict[str, int]
        Dictionary mapping SELFIES tokens to their occurrence counts.
    config : Config
        Configuration object containing min/max token thresholds and
        progress interval.
    stats : Statistics
        Object to record processing counts.
    pbar : tqdm
        tqdm progress bar used for visual feedback.
    progress_counter : Dict[str, int]
        A single-element dictionary storing the number of lines processed
        since the last progress update.  Using a dictionary allows this
        value to be updated within nested scopes.
    """
    for smiles in smiles_iter:
        # Attempt to convert to SELFIES
        try:
            sf: str = selfies.encoder(smiles)  # type: ignore
        except Exception:
            stats.failed += 1
            continue
        # Split into tokens
        tokens = list(selfies.split_selfies(sf))  # type: ignore
        token_count = len(tokens)
        # Filter by length
        if token_count < config.min_tokens:
            stats.too_short += 1
            continue
        if token_count > config.max_tokens:
            stats.too_long += 1
            continue
        # Write to CSV: join tokens with space for token separation
        writer.writerow({"selfies": sf})
        stats.kept += 1
        # Update vocabulary counts
        for tok in tokens:
            vocab[tok] += 1
        # Update progress
        progress_counter["count"] += 1
        if progress_counter["count"] >= config.progress_interval:
            pbar.update(progress_counter["count"])
            progress_counter["count"] = 0
            # Print a log message to stderr to avoid mixing with CSV output
            print(stats.report(), file=sys.stderr)


def build_vocabulary(vocab_counts):
    special_tokens = ["<pad>", "<sos>", "<eos>"]
    tokens = sorted(vocab_counts.keys())
    return special_tokens + tokens



def main(args: Optional[List[str]] = None) -> int:
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Transformer training dataset by converting SMILES to SELFIES "
            "with streaming processing and vocabulary generation."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "Input files or glob patterns to process.  Accepts multiple values."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="transformer_train.csv",
        help="Path to the output CSV file (default: transformer_train.csv)",
    )
    parser.add_argument(
        "--vocab_json",
        type=str,
        default="vocab_transformer.json",
        help="Path to the output vocabulary JSON file (default: vocab_transformer.json)",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=5,
        help="Minimum number of SELFIES tokens required to keep a molecule (default: 5)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum number of SELFIES tokens allowed to keep a molecule (default: 150)",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1_000_000,
        help=(
            "Number of lines processed between progress updates "
            "(default: 1,000,000)"
        ),
    )

    ns = parser.parse_args(args)
    # Expand input patterns into file paths
    input_paths: List[Path] = []
    for pattern in ns.input:
        # Use pathlib for globbing; handle absolute and relative patterns
        p = Path(pattern)
        if p.is_file():
            input_paths.append(p)
        else:
            # Glob the pattern relative to current working directory
            matched = list(Path().glob(pattern))
            if not matched:
                print(f"Warning: pattern '{pattern}' matched no files", file=sys.stderr)
            input_paths.extend([m for m in matched if m.is_file()])
    if not input_paths:
        parser.error("No input files found. Please provide valid file paths or patterns.")

    config = Config(
        input_files=input_paths,
        output_csv=Path(ns.output_csv),
        vocab_json=Path(ns.vocab_json),
        min_tokens=ns.min_tokens,
        max_tokens=ns.max_tokens,
        progress_interval=ns.progress_interval,
    )

    # Create output directory if it doesn't exist
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    config.vocab_json.parent.mkdir(parents=True, exist_ok=True)

    vocab: Dict[str, int] = defaultdict(int)
    stats = Statistics()
    # Counter to track lines since last progress update.  Using a dict allows
    # mutation within nested scopes.
    progress_counter = {"count": 0}

    # Open output CSV and write header
    with config.output_csv.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["selfies"])
        writer.writeheader()
        # Configure tqdm progress bar.  We do not know the total number of lines
        # beforehand, so we leave total as None.  The bar updates after a
        # configurable number of processed lines.
        with tqdm(
            total=None,
            unit="lines",
            dynamic_ncols=True,
            mininterval=1.0,
            desc="Processing",
        ) as pbar:
            # Process each input file in sequence
            for file_path in config.input_files:
                delim = detect_delimiter(file_path)
                try:
                    if delim is not None:
                        smiles_iter = iter_smiles_from_delimited(
                            file_path, delim, stats
                        )
                    else:
                        smiles_iter = iter_smiles_from_text(
                            file_path, stats
                        )
                    process_smiles_stream(
                        smiles_iter,
                        writer,
                        vocab,
                        config,
                        stats,
                        pbar,
                        progress_counter,
                    )
                except FileNotFoundError:
                    print(f"Warning: file '{file_path}' not found", file=sys.stderr)
                    continue
            # Final progress update for any remaining lines
            if progress_counter["count"] > 0:
                pbar.update(progress_counter["count"])
                progress_counter["count"] = 0
                print(stats.report(), file=sys.stderr)

    # Build vocabulary JSON and write to disk
    vocab_mapping = build_vocabulary(vocab)
    with config.vocab_json.open("w", encoding="utf-8") as vocab_f:
        json.dump(vocab_mapping, vocab_f, indent=2)

    # Emit final statistics to stderr
    print("\nFinal statistics:", file=sys.stderr)
    print(stats.report(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())