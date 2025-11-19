"""
Markdown Table Parser - Extract and filter transaction data from markdown tables.

Parses markdown bank statement tables and filters out credit transactions
based on empty Withdrawal column.
"""

import re
from typing import Dict, List


def parse_markdown_table(markdown_text: str) -> List[Dict[str, str]]:
    """
    Parse markdown table into list of row dictionaries.

    Args:
        markdown_text: Markdown table string

    Returns:
        List of dictionaries, one per row (excluding header separator)

    Example:
        >>> table = '''
        ... | Date | Amount |
        ... |------|--------|
        ... | 01/01 | $100 |
        ... '''
        >>> rows = parse_markdown_table(table)
        >>> rows[0]
        {'Date': '01/01', 'Amount': '$100'}
    """
    lines = [line.strip() for line in markdown_text.strip().split("\n") if line.strip()]

    if len(lines) < 2:
        return []

    # First line is headers
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split("|") if h.strip()]

    # Find data rows (skip header separator line with dashes)
    data_rows = []
    for line in lines[1:]:
        # Skip separator lines (contain only |, -, and spaces)
        if re.match(r"^[\|\-\s]+$", line):
            continue

        # Parse row
        cells = [c.strip() for c in line.split("|") if c or c == ""]

        # Handle edge case: line starts/ends with | so split creates empty strings
        # Filter to match header count
        if len(cells) > len(headers):
            cells = cells[1 : len(headers) + 1]  # Remove leading/trailing empty strings

        if len(cells) == len(headers):
            row_dict = dict(zip(headers, cells, strict=False))
            data_rows.append(row_dict)

    return data_rows


def filter_withdrawal_transactions(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filter to keep only withdrawal transactions (rows with non-empty Withdrawal column).

    Args:
        rows: List of row dictionaries from markdown table

    Returns:
        Filtered list containing only rows where Withdrawal column has a value

    Example:
        >>> rows = [
        ...     {'Date': '01/01', 'Description': 'PURCHASE', 'Withdrawal': '$50', 'Deposit': ''},
        ...     {'Date': '02/01', 'Description': 'SALARY', 'Withdrawal': '', 'Deposit': '$2000'},
        ... ]
        >>> filtered = filter_withdrawal_transactions(rows)
        >>> len(filtered)
        1
        >>> filtered[0]['Description']
        'PURCHASE'
    """
    filtered = []

    for row in rows:
        withdrawal = row.get("Withdrawal", "").strip()

        # Keep row if Withdrawal column has content (not empty, not just whitespace)
        if withdrawal and withdrawal != "":
            filtered.append(row)

    return filtered


def extract_transaction_fields(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Extract transaction fields in the format needed for evaluation.

    Args:
        rows: List of filtered transaction row dictionaries

    Returns:
        Dictionary with pipe-separated fields:
        - TRANSACTION_DATES
        - LINE_ITEM_DESCRIPTIONS
        - TRANSACTION_AMOUNTS_PAID

    Example:
        >>> rows = [
        ...     {'Date': '01/01/2025', 'Description': 'PURCHASE', 'Withdrawal': '$50.00'},
        ...     {'Date': '02/01/2025', 'Description': 'ATM', 'Withdrawal': '$100.00'},
        ... ]
        >>> fields = extract_transaction_fields(rows)
        >>> fields['TRANSACTION_DATES']
        '01/01/2025 | 02/01/2025'
    """
    dates = []
    descriptions = []
    amounts = []

    for row in rows:
        dates.append(row.get("Date", "").strip())
        descriptions.append(row.get("Description", "").strip())
        amounts.append(row.get("Withdrawal", "").strip())

    return {
        "TRANSACTION_DATES": " | ".join(dates),
        "LINE_ITEM_DESCRIPTIONS": " | ".join(descriptions),
        "TRANSACTION_AMOUNTS_PAID": " | ".join(amounts),
    }


def process_bank_statement_markdown(markdown_text: str) -> Dict[str, str]:
    """
    Complete pipeline: parse markdown table, filter withdrawals, extract fields.

    Args:
        markdown_text: Markdown table from vision model

    Returns:
        Dictionary with extracted transaction fields

    Example:
        >>> markdown = '''
        ... | Date | Description | Withdrawal | Deposit | Balance |
        ... |------|-------------|------------|---------|---------|
        ... | 01/01 | PURCHASE | $50 | | $950 |
        ... | 02/01 | SALARY | | $2000 | $2950 |
        ... '''
        >>> result = process_bank_statement_markdown(markdown)
        >>> result['TRANSACTION_DATES']
        '01/01'
        >>> result['TRANSACTION_AMOUNTS_PAID']
        '$50'
    """
    # Parse markdown table
    rows = parse_markdown_table(markdown_text)

    # Filter to keep only withdrawals
    withdrawal_rows = filter_withdrawal_transactions(rows)

    # Extract fields
    fields = extract_transaction_fields(withdrawal_rows)

    # Add metadata
    fields["_total_rows"] = len(rows)
    fields["_withdrawal_rows"] = len(withdrawal_rows)
    fields["_filtered_out"] = len(rows) - len(withdrawal_rows)

    return fields
