"""
Table Post-Processor - Clean and validate extracted bank statement tables

Fixes common vision model errors:
1. Column misalignment (debits in credit column, credits in debit column)
2. Balance duplication (Balance value copied to debit/credit columns)
3. Empty cell handling (adds "NOT_FOUND" for consistency)
4. Balance arithmetic validation

Usage:
    >>> from common.table_postprocessor import TablePostProcessor
    >>>
    >>> processor = TablePostProcessor()
    >>> cleaned_table = processor.process_markdown_table(raw_markdown)
    >>> print(f"Fixed {processor.fixes_applied} errors")
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# Optional Rich console for prettier output
try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


@dataclass
class TransactionRow:
    """Represents a single transaction row."""

    date: str
    description: str
    withdrawal: str
    deposit: str
    balance: str
    row_number: int


class TablePostProcessor:
    """Post-processes extracted bank statement tables to fix common errors."""

    # Transaction keywords for identifying withdrawals (debits)
    WITHDRAWAL_KEYWORDS = [
        "eftpos",
        "purchase",
        "payment",
        "atm",
        "withdrawal",
        "debit",
        "bpay",
        "direct debit",
        "mortgage",
        "insurance",
        "utilities",
        "subscription",
        "fee",
        "charge",
        "transfer to",
        "expense",
        "invoice payment",
    ]

    # Transaction keywords for identifying deposits (credits)
    DEPOSIT_KEYWORDS = [
        "salary",
        "pay run",
        "payroll",
        "direct credit",
        "deposit",
        "interest",
        "dividend",
        "refund",
        "centrelink",
        "payment from",
        "transfer from",
        "income",
    ]

    def __init__(self, verbose: bool = True):
        """
        Initialize post-processor.

        Args:
            verbose: Whether to print detailed fix information
        """
        self.verbose = verbose
        self.fixes_applied = 0
        self.errors_detected = []
        self.has_balance_column = False
        self.is_reverse_chronological = False

    def process_markdown_table(self, markdown_table: str) -> str:
        """
        Process and clean a markdown table.

        Args:
            markdown_table: Raw markdown table string

        Returns:
            Cleaned markdown table string
        """
        # Reset counters
        self.fixes_applied = 0
        self.errors_detected = []

        # Parse markdown table
        rows = self._parse_markdown_table(markdown_table)

        if not rows:
            if self.verbose:
                self._print("⚠️  No rows to process")
            return markdown_table

        # Detect if balance column exists (check if any row has non-empty balance)
        self.has_balance_column = any(
            row.balance and row.balance.strip() for row in rows
        )

        if not self.has_balance_column:
            if self.verbose:
                self._print("ℹ️  No balance column detected - skipping balance validation")
            # Still apply keyword-based fixes, but skip balance validation
        else:
            # Detect chronological order by comparing first and last dates
            self.is_reverse_chronological = self._detect_chronological_order(rows)
            order_str = "reverse chronological (newest first)" if self.is_reverse_chronological else "chronological (oldest first)"
            if self.verbose:
                self._print(f"ℹ️  Table order: {order_str}")

        # Process each row
        cleaned_rows = []
        for row in rows:
            cleaned_row = self._process_row(row, rows)
            cleaned_rows.append(cleaned_row)

        # Convert back to markdown
        cleaned_markdown = self._rows_to_markdown(cleaned_rows)

        # Print summary
        if self.verbose:
            self._print(f"\n✅ Post-processing complete: {self.fixes_applied} fixes applied")
            if self.errors_detected:
                self._print(f"⚠️  {len(self.errors_detected)} errors detected:")
                for error in self.errors_detected[:5]:  # Show first 5
                    self._print(f"  - {error}")
                if len(self.errors_detected) > 5:
                    self._print(f"  ... and {len(self.errors_detected) - 5} more")

        return cleaned_markdown

    def _parse_markdown_table(self, markdown_table: str) -> List[TransactionRow]:
        """
        Parse markdown table into TransactionRow objects.

        Handles 3, 4, or 5 column tables:
        - 3 columns: Date | Description | Amount
        - 4 columns: Date | Description | Withdrawal | Deposit (no balance)
        - 5 columns: Date | Description | Withdrawal | Deposit | Balance
        """
        rows = []
        lines = markdown_table.strip().split("\n")

        # Skip header and separator rows
        data_lines = [line for line in lines[2:] if line.strip() and line.startswith("|")]

        for i, line in enumerate(data_lines, start=1):
            # Split by pipe and clean
            cells = [cell.strip() for cell in line.split("|")[1:-1]]

            # Parse based on column count
            if len(cells) == 3:
                # 3-column: Date | Description | Amount
                rows.append(
                    TransactionRow(
                        date=cells[0],
                        description=cells[1],
                        withdrawal=cells[2],  # Single amount column
                        deposit="",
                        balance="",
                        row_number=i,
                    )
                )
            elif len(cells) == 4:
                # 4-column: Date | Description | Withdrawal | Deposit (no balance)
                rows.append(
                    TransactionRow(
                        date=cells[0],
                        description=cells[1],
                        withdrawal=cells[2],
                        deposit=cells[3],
                        balance="",  # No balance column
                        row_number=i,
                    )
                )
            elif len(cells) >= 5:
                # 5-column: Date | Description | Withdrawal | Deposit | Balance
                rows.append(
                    TransactionRow(
                        date=cells[0],
                        description=cells[1],
                        withdrawal=cells[2],
                        deposit=cells[3],
                        balance=cells[4],
                        row_number=i,
                    )
                )

        return rows

    def _process_row(
        self, row: TransactionRow, all_rows: List[TransactionRow]
    ) -> TransactionRow:
        """
        Process a single row to detect and fix errors.

        Args:
            row: Transaction row to process
            all_rows: All rows (for balance validation)

        Returns:
            Cleaned TransactionRow
        """
        # Make a copy to modify
        withdrawal = row.withdrawal.strip()
        deposit = row.deposit.strip()
        balance = row.balance.strip()
        description = row.description.lower()

        # Check if balance validation is available
        # (requires balance column AND chronologically previous transaction exists)
        use_balance_validation = (
            self.has_balance_column
            and self._has_chronologically_previous_transaction(row, all_rows)
        )

        # Step 1: Check for Balance duplication (only if balance column exists)
        balance_amount = self._extract_amount(balance)
        withdrawal_amount = self._extract_amount(withdrawal) if withdrawal else None
        deposit_amount = self._extract_amount(deposit) if deposit else None

        # Fix Balance duplication in withdrawal column
        if withdrawal_amount and balance_amount and withdrawal_amount == balance_amount:
            self._log_fix(
                row.row_number,
                f"Balance duplication detected in Withdrawal: {withdrawal} = {balance}",
            )
            withdrawal = ""
            self.fixes_applied += 1

        # Fix Balance duplication in deposit column
        if deposit_amount and balance_amount and deposit_amount == balance_amount:
            self._log_fix(
                row.row_number,
                f"Balance duplication detected in Deposit: {deposit} = {balance}",
            )
            deposit = ""
            self.fixes_applied += 1

        # Step 2: Identify transaction type from description
        is_withdrawal_desc = any(
            keyword in description for keyword in self.WITHDRAWAL_KEYWORDS
        )
        is_deposit_desc = any(keyword in description for keyword in self.DEPOSIT_KEYWORDS)

        # Step 3: Check column alignment
        has_withdrawal = bool(withdrawal and withdrawal.strip())
        has_deposit = bool(deposit and deposit.strip())

        # If both columns have values, this is likely an error
        if has_withdrawal and has_deposit:
            self._log_fix(
                row.row_number,
                f"Both columns populated: W={withdrawal}, D={deposit}. Using keywords to fix.",
            )

            if is_withdrawal_desc and not is_deposit_desc:
                # Keep withdrawal, clear deposit
                deposit = ""
                self.fixes_applied += 1
            elif is_deposit_desc and not is_withdrawal_desc:
                # Keep deposit, clear withdrawal
                withdrawal = ""
                self.fixes_applied += 1
            else:
                # Use Balance change direction as tiebreaker (only if balance validation enabled)
                if use_balance_validation:
                    prev_balance = self._get_previous_balance(row, all_rows)
                    if prev_balance and balance_amount:
                        # Account for chronological order
                        if self.is_reverse_chronological:
                            # prev_balance is chronologically AFTER current
                            # If current < prev → balance INCREASED → deposit
                            if balance_amount < prev_balance:
                                withdrawal = ""  # Balance increased = deposit
                            else:
                                deposit = ""     # Balance decreased = withdrawal
                        else:
                            # prev_balance is chronologically BEFORE current
                            # If current < prev → balance DECREASED → withdrawal
                            if balance_amount < prev_balance:
                                deposit = ""     # Balance decreased = withdrawal
                            else:
                                withdrawal = ""  # Balance increased = deposit
                        self.fixes_applied += 1

        # Step 4: Check if value is in wrong column based on keywords
        elif has_withdrawal and not has_deposit and is_deposit_desc:
            # Withdrawal column has value but description indicates deposit
            self._log_fix(
                row.row_number,
                f"Misaligned: '{row.description}' is DEPOSIT but value in Withdrawal",
            )
            deposit = withdrawal
            withdrawal = ""
            self.fixes_applied += 1

        elif has_deposit and not has_withdrawal and is_withdrawal_desc:
            # Deposit column has value but description indicates withdrawal
            self._log_fix(
                row.row_number,
                f"Misaligned: '{row.description}' is WITHDRAWAL but value in Deposit",
            )
            withdrawal = deposit
            deposit = ""
            self.fixes_applied += 1

        # Step 5: Add "NOT_FOUND" for empty cells (optional - can be disabled)
        # withdrawal = withdrawal if withdrawal.strip() else "NOT_FOUND"
        # deposit = deposit if deposit.strip() else "NOT_FOUND"

        return TransactionRow(
            date=row.date,
            description=row.description,
            withdrawal=withdrawal,
            deposit=deposit,
            balance=balance,
            row_number=row.row_number,
        )

    def _extract_amount(self, value: str) -> Optional[float]:
        """Extract numeric amount from a string like '$1,234.56'."""
        if not value or not value.strip():
            return None

        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r"[^\d.]", "", value)

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _detect_chronological_order(self, rows: List[TransactionRow]) -> bool:
        """
        Detect if table is in reverse chronological order.

        Args:
            rows: All transaction rows

        Returns:
            True if reverse chronological (newest first), False if chronological (oldest first)
        """
        if len(rows) < 2:
            return False

        first_date = self._parse_date(rows[0].date)
        last_date = self._parse_date(rows[-1].date)

        if not first_date or not last_date:
            # Can't determine order, assume chronological
            return False

        # If first date > last date, it's reverse chronological
        return first_date > last_date

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str or not date_str.strip():
            return None

        # Common date formats in bank statements
        date_formats = [
            "%d/%m/%Y",  # 07/09/2025
            "%d-%m-%Y",  # 07-09-2025
            "%Y-%m-%d",  # 2025-09-07
            "%d %b %Y",  # 07 Sep 2025
            "%d %B %Y",  # 07 September 2025
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

    def _has_chronologically_previous_transaction(
        self, row: TransactionRow, all_rows: List[TransactionRow]
    ) -> bool:
        """
        Check if chronologically previous transaction exists.

        Args:
            row: Current transaction row
            all_rows: All transaction rows

        Returns:
            True if chronologically previous transaction exists
        """
        if self.is_reverse_chronological:
            # Reverse chronological: previous = next row in list
            return row.row_number < len(all_rows)
        else:
            # Forward chronological: previous = previous row in list
            return row.row_number > 1

    def _get_previous_balance(
        self, row: TransactionRow, all_rows: List[TransactionRow]
    ) -> Optional[float]:
        """
        Get the balance from the CHRONOLOGICALLY PREVIOUS transaction.

        CRITICAL: Finds the transaction that happened EARLIER IN TIME,
        regardless of table position. Balance validation requires
        chronological order.

        Args:
            row: Current transaction row
            all_rows: All transaction rows

        Returns:
            Balance from chronologically previous transaction, or None
        """
        if self.is_reverse_chronological:
            # Reverse chronological: newer dates at top
            # Chronologically previous = NEXT row in list (happened earlier)
            if row.row_number < len(all_rows):
                prev_row = all_rows[row.row_number]  # Next row (0-indexed)
            else:
                return None  # Last row, no chronologically previous
        else:
            # Forward chronological: older dates at top
            # Chronologically previous = PREVIOUS row in list (happened earlier)
            if row.row_number > 1:
                prev_row = all_rows[row.row_number - 2]  # Previous row (0-indexed)
            else:
                return None  # First row, no chronologically previous

        return self._extract_amount(prev_row.balance)

    def _print(self, message: str):
        """Print message using Rich if available, otherwise plain print."""
        if HAS_RICH and console:
            console.print(message)
        else:
            print(message)

    def _log_fix(self, row_number: int, message: str):
        """Log a fix that was applied."""
        error_msg = f"Row {row_number}: {message}"
        self.errors_detected.append(error_msg)

        if self.verbose:
            self._print(f"  🔧 Fix applied - {error_msg}")

    def _rows_to_markdown(self, rows: List[TransactionRow]) -> str:
        """Convert TransactionRow objects back to markdown table."""
        lines = [
            "| Date | Description | Withdrawal | Deposit | Balance |",
            "|------|-------------|------------|---------|---------|",
        ]

        for row in rows:
            line = (
                f"| {row.date} | {row.description} | {row.withdrawal} | "
                f"{row.deposit} | {row.balance} |"
            )
            lines.append(line)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Test with sample markdown table
    sample_table = """| Date | Description | Withdrawal | Deposit | Balance |
|------|-------------|------------|---------|---------|
| 07/09/2025 | EFTPOS Cash Out PRICELINE PHARMACY | $322.18 |  | $48890.58 |
| 02/09/2025 | Salary Payment ATO | $8105.71 |  | $51511.48 |
| 01/09/2025 | DD INSURANCE ACME CORP |  | $43405.77 | $43405.77 |"""

    processor = TablePostProcessor(verbose=True)
    cleaned = processor.process_markdown_table(sample_table)
    print("\n" + "=" * 80)
    print("CLEANED TABLE:")
    print("=" * 80)
    print(cleaned)
