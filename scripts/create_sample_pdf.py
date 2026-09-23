"""Create a sample employee handbook PDF for testing ingestion."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path


def create_sample_pdf(output_path: Path) -> None:
    """Generate a multi-page test PDF with realistic content."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    # ---- Page 1 ----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Employee Handbook")
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 1.3 * inch, "Acme Corporation — Version 3.2")
    c.drawString(1 * inch, height - 1.5 * inch, "Effective: January 2024")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, height - 2.2 * inch, "Section 1: Introduction")
    c.setFont("Helvetica", 11)

    text_y = height - 2.5 * inch
    lines = [
        "Welcome to Acme Corporation. This handbook outlines the policies",
        "and procedures that govern your employment. Please read it carefully",
        "and refer to it whenever you have questions about company policy.",
        "",
        "The information in this handbook is subject to change. The most",
        "current version is always available on the company intranet.",
    ]
    for line in lines:
        c.drawString(1 * inch, text_y, line)
        text_y -= 0.2 * inch

    # Add a fake page footer
    c.setFont("Helvetica", 8)
    c.drawString(width / 2 - 0.5 * inch, 0.5 * inch, "Page 1 of 3")

    c.showPage()

    # ---- Page 2 ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, height - 1 * inch, "Section 4.2: Vacation Policy")

    c.setFont("Helvetica", 11)
    text_y = height - 1.4 * inch
    lines = [
        "Employees receive 15 days of paid annual leave per calendar year.",
        "Leave accrues monthly at 1.25 days per month. Unused leave may be",
        "carried over to the following year, up to a maximum of 5 days.",
        "",
        "Requests for vacation must be submitted at least two weeks in advance",
        "through the HR portal. Approval is subject to manager discretion and",
        "business needs. During peak periods, requests may be denied.",
        "",
        "Employees who leave the company will be paid out for accrued but",
        "unused vacation days in their final paycheck.",
    ]
    for line in lines:
        c.drawString(1 * inch, text_y, line)
        text_y -= 0.2 * inch

    c.setFont("Helvetica", 8)
    c.drawString(width / 2 - 0.5 * inch, 0.5 * inch, "Page 2 of 3")

    c.showPage()

    # ---- Page 3 ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, height - 1 * inch, "Section 7.1: Password Policy")

    c.setFont("Helvetica", 11)
    text_y = height - 1.4 * inch
    lines = [
        "All company accounts must use passwords that meet the following",
        "requirements: minimum 12 characters, at least one uppercase letter,",
        "one lowercase letter, one number, and one special character.",
        "",
        "Passwords must be changed every 90 days. Reuse of the previous",
        "five passwords is prohibited. Multi-factor authentication (MFA)",
        "is required for all accounts with access to production systems.",
        "",
        "Suspected password compromise must be reported to security@acme.com",
        "within 24 hours.",
    ]
    for line in lines:
        c.drawString(1 * inch, text_y, line)
        text_y -= 0.2 * inch

    c.setFont("Helvetica", 8)
    c.drawString(width / 2 - 0.5 * inch, 0.5 * inch, "Page 3 of 3")

    c.save()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    output = Path("data/raw/employee_handbook.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    create_sample_pdf(output)