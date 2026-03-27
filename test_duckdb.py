#!/usr/bin/env python
"""Test DuckDB SQL feasibility checks from notebook 07"""

import duckdb
import pandas as pd

print("\n" + "="*60)
print("HANDS-ON TEST: DuckDB SQL Feasibility Checks")
print("="*60)

con = duckdb.connect()
con.execute("CREATE VIEW master AS SELECT * FROM read_parquet('data/processed/master_patient_table.parquet')")

# Feasibility Check 1: Subgroup availability
print("\n[Check 1] Subgroup Data Availability (Age 60+)")
print("-" * 60)
result = con.execute("""
    SELECT 
        source_dataset,
        CASE WHEN age >= 60 THEN '60+' ELSE 'Under 60' END AS age_group,
        COUNT(*) AS patient_count,
        ROUND(AVG(diabetes_label), 3) AS diabetes_rate
    FROM master
    WHERE diabetes_label IS NOT NULL
    GROUP BY source_dataset, age_group
    ORDER BY source_dataset, age_group
""").df()
print(result.to_string(index=False))

# Feasibility Check 3: Class balance
print("\n[Check 3] Label Class Balance")
print("-" * 60)
result = con.execute("""
    SELECT
        source_dataset,
        CAST(diabetes_label AS INTEGER) AS label,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY source_dataset), 1) AS pct
    FROM master
    WHERE diabetes_label IS NOT NULL
    GROUP BY source_dataset, diabetes_label
    ORDER BY source_dataset, label
""").df()
print(result.to_string(index=False))

# Feasibility Check 4: Cross-cycle biomarker shift
print("\n[Check 4] Cross-Cycle Biomarker Shift (NHANES)")
print("-" * 60)
result = con.execute("""
    SELECT
        nhanes_cycle,
        ROUND(AVG(glucose), 2)    AS avg_glucose,
        ROUND(AVG(bmi), 2)        AS avg_bmi,
        ROUND(AVG(hba1c), 2)      AS avg_hba1c,
        COUNT(*) AS n
    FROM master
    WHERE source_dataset = 'nhanes'
    GROUP BY nhanes_cycle
""").df()
print(result.to_string(index=False))

# Generate feasibility summary
print("\n[Summary] Master Table Overview")
print("-" * 60)
summary = con.execute("""
    SELECT 
        'Project P.U.L.S.E.' AS project,
        COUNT(*) AS total_patients,
        COUNT(DISTINCT source_dataset) AS datasets,
        ROUND(COUNT(diabetes_label) * 100.0 / COUNT(*), 1) AS diabetes_label_coverage_pct,
        ROUND(AVG(age), 1) AS avg_age,
        ROUND(AVG(bmi), 1) AS avg_bmi
    FROM master
""").df()
print(summary.to_string(index=False))

print("\n" + "="*60)
print("✓ DuckDB SQL queries executed successfully!")
print("="*60 + "\n")
