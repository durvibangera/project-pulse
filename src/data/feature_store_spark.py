# src/data/feature_store_spark.py
# PySpark replication of the PULSE feature store.
# Mirrors feature_store.py logic using Spark DataFrames for large-scale processing.
# The original pandas version (feature_store.py) is left untouched.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, IntegerType
import os


def get_spark():
    return (
        SparkSession.builder
        .appName("PULSE_FeatureStore")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_nhanes_spark(spark, cycle_dir: str, cycle: str):
    """
    Loads NHANES XPT files via pandas bridge (pyreadstat),
    then converts to Spark DataFrame for large-scale processing.
    """
    import pyreadstat
    import pandas as pd

    frames = {}
    for fname in os.listdir(cycle_dir):
        lower = fname.lower()
        fpath = os.path.join(cycle_dir, fname)
        if "demo" in lower:
            df, _ = pyreadstat.read_xport(fpath)
            frames["demo"] = df
        elif "ghb" in lower:
            df, _ = pyreadstat.read_xport(fpath)
            frames["ghb"] = df
        elif "bpx" in lower:
            df, _ = pyreadstat.read_xport(fpath)
            frames["bpx"] = df

    merged = frames["demo"]
    for key in ["ghb", "bpx"]:
        if key in frames:
            merged = merged.merge(frames[key], on="SEQN", how="left")

    # Convert to Spark
    sdf = spark.createDataFrame(merged)
    return sdf


def clean_nhanes_spark(sdf, cycle: str):
    return sdf.select(
        F.concat(F.col("SEQN").cast(StringType()), F.lit(f"_{cycle}")).alias("patient_id"),
        F.col("RIDAGEYR").cast(DoubleType()).alias("age"),
        F.when(F.col("RIAGENDR") == 1, 1).when(F.col("RIAGENDR") == 2, 0).alias("gender"),
        F.col("BMXBMI").cast(DoubleType()).alias("bmi"),
        F.col("LBXGLU").cast(DoubleType()).alias("glucose"),
        F.col("LBXGH").cast(DoubleType()).alias("hba1c"),
        F.col("BPXSY1").cast(DoubleType()).alias("blood_pressure_sys"),
        F.col("BPXDI1").cast(DoubleType()).alias("blood_pressure_dia"),
        F.when(
            (F.col("LBXGH") >= 6.5) | (F.col("LBXGLU") >= 126), 1
        ).otherwise(0).alias("diabetes_label"),
        F.lit(None).cast(DoubleType()).alias("cardio_label"),
        F.lit("nhanes").alias("source_dataset"),
        F.lit(cycle).alias("nhanes_cycle"),
    )


def build_feature_store_spark():
    spark = get_spark()

    # NOTE: actual raw dirs use hyphens (2015-16, 2021-23) matching the filesystem
    nhanes_2015_raw = load_nhanes_spark(spark, "data/raw/nhanes/2015-16", "2015_16")
    nhanes_2015 = clean_nhanes_spark(nhanes_2015_raw, "2015_16")

    nhanes_2021_raw = load_nhanes_spark(spark, "data/raw/nhanes/2021-23", "2021_23")
    nhanes_2021 = clean_nhanes_spark(nhanes_2021_raw, "2021_23")

    combined = nhanes_2015.unionByName(nhanes_2021, allowMissingColumns=True)

    # Write to Parquet (Spark-native, partitioned by cycle for performance)
    combined.write.mode("overwrite") \
        .partitionBy("nhanes_cycle") \
        .parquet("data/processed/nhanes_spark_partitioned/")

    print(f"[PULSE Spark] Written {combined.count()} NHANES records.")

    # Print schema for documentation
    combined.printSchema()
    combined.groupBy("nhanes_cycle").count().show()

    spark.stop()


if __name__ == "__main__":
    build_feature_store_spark()
