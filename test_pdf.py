#!/usr/bin/env python
"""Test PDF report generation"""

import os

print("\n" + "="*60)
print("TEST: PDF Report Generation")
print("="*60)

from src.reporting.pdf_report import generate_model_report

try:
    print("\nGenerating PDF report...")
    generate_model_report(output_path="reports/PULSE_Model_Report_TEST.pdf")
    
    if os.path.exists("reports/PULSE_Model_Report_TEST.pdf"):
        size_kb = os.path.getsize("reports/PULSE_Model_Report_TEST.pdf") / 1024
        print(f"\n✓ PDF generated successfully")
        print(f"  File: reports/PULSE_Model_Report_TEST.pdf")
        print(f"  Size: {size_kb:.1f} KB")
    else:
        print("\n✗ PDF file not found after generation")
except Exception as e:
    print(f"\n✗ PDF generation failed: {e}")
    import traceback
    traceback.print_exc()

print("="*60 + "\n")
