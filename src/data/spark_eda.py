# src/data/spark_eda.py
# Demonstrates Spark SQL analytics capability over the PULSE master Parquet table.
# Run: python src/data/spark_eda.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def run_spark_eda():
    spark = SparkSession.builder.appName("PULSE_EDA").getOrCreate()

    # Read the master Parquet file into Spark
    df = spark.read.parquet("data/processed/master_patient_table.parquet")
    df.createOrReplaceTempView("master")

    print("\n=== Spark EDA 1: Patient counts by dataset ===")
    spark.sql("""
        SELECT source_dataset, COUNT(*) as n,
               ROUND(AVG(age), 1) as avg_age,
               ROUND(AVG(bmi), 1) as avg_bmi
        FROM master GROUP BY source_dataset ORDER BY n DESC
    """).show()

    print("=== Spark EDA 2: Diabetes prevalence by age group ===")
    spark.sql("""
        SELECT
            FLOOR(age / 10) * 10 AS age_decade,
            COUNT(*) AS n,
            ROUND(AVG(diabetes_label), 3) AS diabetes_rate
        FROM master
        WHERE diabetes_label IS NOT NULL AND age IS NOT NULL
        GROUP BY age_decade ORDER BY age_decade
    """).show()

    print("=== Spark EDA 3: Cross-cycle biomarker summary ===")
    spark.sql("""
        SELECT nhanes_cycle,
               ROUND(AVG(glucose), 2) AS avg_glucose,
               ROUND(AVG(hba1c), 2) AS avg_hba1c,
               ROUND(AVG(bmi), 2) AS avg_bmi
        FROM master
        WHERE source_dataset = 'nhanes'
        GROUP BY nhanes_cycle
    """).show()

    spark.stop()


if __name__ == "__main__":
    run_spark_eda()
