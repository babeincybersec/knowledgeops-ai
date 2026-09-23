"""Clean extracted PDF text by removing common noise."""

import re


# Patterns that indicate noise

# Matches lines like "Page 1 of 3", "Page 12", "- 3 -", "1/3"
PAGE_FOOTER_PATTERN = re.compile(
    r"^\s*"
    r"(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|-\s*\d+\s*-)"
    r"\s*$",
    re.IGNORECASE,
)

# Matches a line that is just a number (likely a page number)
STANDALONE_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*$")

# Matches 3+ consecutive newlines (collapse to 2)
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")

# Matches two or more spaces (collapse to one)
EXCESS_SPACES_PATTERN = re.compile(r" {2,}")

# Matches trailing whitespace on lines
TRAILING_WHITESPACE_PATTERN = re.compile(r"[ \t]+\n")

# Matches hyphenated line breaks: "vaca-\ntion" → "vacation"
HYPHENATION_PATTERN = re.compile(r"(\w+)-\n(\w+)")


def remove_page_footers(text: str) -> str:
    """Remove lines that look like page footers or standalone page numbers."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if PAGE_FOOTER_PATTERN.match(line):
            continue
        if STANDALONE_NUMBER_PATTERN.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def fix_hyphenation(text: str) -> str:
    """
    Rejoin words split across lines with a hyphen.

    Example: "vaca-\\ntion" becomes "vacation".
    """
    return HYPHENATION_PATTERN.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive spaces and newlines, strip trailing whitespace."""
    text = TRAILING_WHITESPACE_PATTERN.sub("\n", text)
    text = EXCESS_SPACES_PATTERN.sub(" ", text)
    text = EXCESS_NEWLINES_PATTERN.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Run the full cleaning pipeline on a block of text."""
    text = remove_page_footers(text)
    text = fix_hyphenation(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    # Quick test with synthetic input
    sample = """Employee Handbook
Acme Corporation — Version 3.2


Page 1 of 3

Welcome to Acme Corporation. This handbook outlines the
policies and procedures that govern your employ-
ment. Please read it carefully.

12

Section 4.2: Vacation Policy
"""
    print("BEFORE:")
    print(repr(sample))
    print("\nAFTER:")
    print(repr(clean_text(sample)))