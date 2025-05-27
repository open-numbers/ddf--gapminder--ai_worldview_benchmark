#!/usr/bin/env python3
"""
datapoints.py

A script to process CSV files and generate DDF datapoint files.
This script focuses only on the datapoint processing part of etl_notebook.py
and allows specifying which folder to read CSV files from.

Requirements:
    - polars: For data processing (pip install polars)

Usage:
    python datapoints.py --source <source_folder> --output <output_dir>

Example:
    python datapoints.py --source ../source/results --output ../../output
"""

import argparse
import os
import sys
from pathlib import Path

# Check for required dependencies
try:
    import polars as pl
except ImportError:
    print("Error: This script requires the 'polars' package.")
    print("Please install it using: pip install polars")
    sys.exit(1)


def process_datapoints(source_folder, output_dir):
    """
    Process CSV files from the specified source folder and generate DDF datapoint files.

    Args:
        source_folder (str): Path to the folder containing CSV files to process
        output_dir (str): Path to the directory where output files will be saved
    """
    print(f"Reading CSV files from: {source_folder}")

    # Create glob pattern for all CSV files in the source folder
    csv_pattern = os.path.join(source_folder, "*.csv")

    # Read all CSV files in the source folder
    master_output = pl.read_csv(
        csv_pattern,
        schema_overrides={"question_id": pl.Utf8},
    )

    print(f"Found {master_output.height} rows in total")

    # Convert column names to lowercase and connect with underscore
    master_output = master_output.rename(
        {col: col.lower().replace(" ", "_") for col in master_output.columns}
    )

    # Check for multiple dates per combination
    date_check = (
        master_output.group_by(
            ["question_id", "prompt_variation_id", "model_configuration_id"]
        )
        .agg(
            [
                pl.col("last_evaluation_datetime").n_unique().alias("unique_dates"),
                pl.col("last_evaluation_datetime").alias("dates"),
            ]
        )
        .filter(pl.col("unique_dates") > 1)
        .sort("unique_dates", descending=True)
    )

    print("Combinations with multiple dates:")
    print(date_check)

    # Keep only the latest datapoint for each combination
    master_output = (
        master_output.sort(
            [
                "question_id",
                "prompt_variation_id",
                "model_configuration_id",
                "last_evaluation_datetime",
            ]
        )
        .group_by(["question_id", "prompt_variation_id", "model_configuration_id"])
        .agg(pl.all().last())  # Keep the last (most recent) record for each group
    )

    # Now we can remove the language and date columns as they're no longer needed
    master_output = master_output.drop(["language", "last_evaluation_datetime"])

    print(f"After removing duplicates: {master_output.height} rows")

    # Create raw results datapoints with renamed columns
    raw_result_datapoints = master_output.select(
        [
            pl.col("question_id").alias("question"),
            pl.col("model_configuration_id").alias("model_configuration"),
            pl.col("prompt_variation_id").alias("prompt_variation"),
            pl.col("result").alias("evaluation_result"),
        ]
    ).sort(["model_configuration", "question", "prompt_variation"])

    # Save raw results
    output_path = os.path.join(output_dir, "ddf--datapoints--evaluation_result--by--question--model_configuration--prompt_variation.csv")
    raw_result_datapoints.write_csv(output_path)
    print(f"Saved raw results to: {output_path}")

    # Calculate all rates including indecisive
    rates = (
        master_output.group_by(["question_id", "model_configuration_id"])
        .agg(
            [
                pl.col("result").count().alias("total_all_answers"),
                (pl.col("result") == "correct").sum().alias("correct_count"),
                (pl.col("result") == "wrong").sum().alias("wrong_count"),
                (pl.col("result") == "very_wrong").sum().alias("very_wrong_count"),
                (pl.col("result").is_in(["fail", "n/a"])).sum().alias("indecisive_count"),
            ]
        )
        .with_columns(
            [
                # Calculate rates for decisive answers (excluding fail/n/a)
                (
                    pl.when(pl.col("total_all_answers") == pl.col("indecisive_count"))
                    .then(0.0)
                    .otherwise(
                        pl.col("correct_count")
                        / (pl.col("total_all_answers") - pl.col("indecisive_count"))
                        * 100
                    )
                    .alias("correct_rate")
                ),
                (
                    pl.when(pl.col("total_all_answers") == pl.col("indecisive_count"))
                    .then(0.0)
                    .otherwise(
                        pl.col("wrong_count")
                        / (pl.col("total_all_answers") - pl.col("indecisive_count"))
                        * 100
                    )
                    .alias("wrong_rate")
                ),
                (
                    pl.when(pl.col("total_all_answers") == pl.col("indecisive_count"))
                    .then(0.0)
                    .otherwise(
                        pl.col("very_wrong_count")
                        / (pl.col("total_all_answers") - pl.col("indecisive_count"))
                        * 100
                    )
                    .alias("very_wrong_rate")
                ),
                # Calculate indecisive rate from total answers
                (pl.col("indecisive_count") / pl.col("total_all_answers") * 100).alias(
                    "indecisive_rate"
                ),
            ]
        )
        .sort(["question_id", "model_configuration_id"])
    )

    print("\nAll rates calculated")

    # Check for NaN values in rates
    nan_rows = rates.filter(
        pl.col("correct_rate").is_nan()
        | pl.col("wrong_rate").is_nan()
        | pl.col("very_wrong_rate").is_nan()
        | pl.col("indecisive_rate").is_nan()
    )

    if nan_rows.height > 0:
        print(f"\nWARNING: Found {nan_rows.height} rows with NaN values")

    # Rename columns and drop unnecessary ones before saving
    rates = rates.rename(
        {"question_id": "question", "model_configuration_id": "model_configuration"}
    ).drop(
        [
            "total_all_answers",
            "correct_count",
            "wrong_count",
            "very_wrong_count",
            "indecisive_count",
        ]
    )

    # Create and save separate CSV files for each rate
    for rate_type in ["correct_rate", "wrong_rate", "very_wrong_rate", "indecisive_rate"]:
        output_path = os.path.join(output_dir, f"ddf--datapoints--{rate_type}--by--question--model_configuration.csv")
        rates.select(["question", "model_configuration", rate_type]).write_csv(output_path)
        print(f"Saved {rate_type} to: {output_path}")

    # Create the average correct rate for each model
    avg_rates = (
        rates.group_by("model_configuration")
        .agg(pl.col("correct_rate").mean().alias("average_correct_rate"))
        .with_columns(pl.col("average_correct_rate").round(0).cast(pl.Int32))
        .sort("model_configuration")
    )

    output_path = os.path.join(output_dir, "ddf--datapoints--average_correct_rate--by--model_configuration.csv")
    avg_rates.write_csv(output_path)
    print(f"Saved average correct rate to: {output_path}")

    # Create aggregated rates by prompt variation
    rates_by_prompt = (
        master_output.group_by(["model_configuration_id", "prompt_variation_id"])
        .agg(
            [
                pl.col("result").count().alias("total_all_answers"),
                (pl.col("result") == "correct").sum().alias("correct_count"),
                (pl.col("result") == "wrong").sum().alias("wrong_count"),
                (pl.col("result") == "very_wrong").sum().alias("very_wrong_count"),
                (pl.col("result").is_in(["fail", "n/a"])).sum().alias("indecisive_count"),
            ]
        )
        .with_columns(
            [
                # Calculate rates for decisive answers (excluding fail/n/a)
                (
                    pl.when(pl.col("total_all_answers") == pl.col("indecisive_count"))
                    .then(0.0)
                    .otherwise(
                        pl.col("correct_count")
                        / (pl.col("total_all_answers") - pl.col("indecisive_count"))
                        * 100
                    )
                    .alias("correct_rate")
                ),
                (
                    pl.when(pl.col("total_all_answers") == pl.col("indecisive_count"))
                    .then(0.0)
                    .otherwise(
                        pl.col("wrong_count")
                        / (pl.col("total_all_answers") - pl.col("indecisive_count"))
                        * 100
                    )
                    .alias("wrong_rate")
                ),
                (
                    pl.when(pl.col("total_all_answers") == pl.col("indecisive_count"))
                    .then(0.0)
                    .otherwise(
                        pl.col("very_wrong_count")
                        / (pl.col("total_all_answers") - pl.col("indecisive_count"))
                        * 100
                    )
                    .alias("very_wrong_rate")
                ),
                # Calculate indecisive rate from total answers
                (pl.col("indecisive_count") / pl.col("total_all_answers") * 100).alias(
                    "indecisive_rate"
                ),
            ]
        )
        .sort(["model_configuration_id", "prompt_variation_id"])
    )

    # Rename columns and drop unnecessary ones before saving
    rates_by_prompt = rates_by_prompt.rename(
        {
            "model_configuration_id": "model_configuration",
            "prompt_variation_id": "prompt_variation",
        }
    ).drop(
        [
            "total_all_answers",
            "correct_count",
            "wrong_count",
            "very_wrong_count",
            "indecisive_count",
        ]
    )

    # Create and save separate CSV files for each rate by prompt variation
    for rate_type in ["correct_rate", "wrong_rate", "very_wrong_rate", "indecisive_rate"]:
        output_path = os.path.join(output_dir, f"ddf--datapoints--{rate_type}--by--model_configuration--prompt_variation.csv")
        rates_by_prompt.select(
            ["model_configuration", "prompt_variation", rate_type]
        ).write_csv(output_path)
        print(f"Saved {rate_type} by prompt variation to: {output_path}")

    # Get the top and bottom rates for each model
    rates_by_prompt_correct = rates_by_prompt.select(
        ["model_configuration", "prompt_variation", "correct_rate"]
    )

    # Group by model_configuration, sort correct rate and add a rank column
    bottom = rates_by_prompt_correct.group_by("model_configuration").map_groups(
        lambda x: x.sort("correct_rate").head(5)
    ).group_by("model_configuration").agg(
        pl.col("correct_rate").mean()
    ).sort(
        ["model_configuration"]
    )

    top = rates_by_prompt_correct.group_by("model_configuration").map_groups(
        lambda x: x.sort("correct_rate").tail(5)
    ).group_by("model_configuration").agg(
        pl.col("correct_rate").mean()
    ).sort(
        ["model_configuration"]
    )

    # Save top and bottom rates to the source folder to avoid overwriting other files
    current_dir = os.path.dirname(os.path.abspath(__file__))

    bottom_path = os.path.join(current_dir, "bottom_rates.csv")
    bottom.write_csv(bottom_path)
    print(f"Saved bottom rates to: {bottom_path}")

    top_path = os.path.join(current_dir, "top_rates.csv")
    top.write_csv(top_path)
    print(f"Saved top rates to: {top_path}")

    print("\nDatapoint processing complete!")


def main():
    """
    Main entry point for the script.
    Parses command line arguments and runs the datapoint processing.
    """
    parser = argparse.ArgumentParser(
        description="Process CSV files and generate DDF datapoint files."
    )

    parser.add_argument(
        "--source", "-s",
        dest="source_folder",
        default="../source/results",
        help="Path to the folder containing CSV files to process (default: ../source/results)"
    )

    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        default="../..",
        help="Path to the directory where output files will be saved (default: ../..)"
    )

    args = parser.parse_args()

    # Convert to absolute path if relative
    source_folder = os.path.abspath(args.source_folder)

    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return 1

    # Convert to absolute path if relative
    output_dir = os.path.abspath(args.output_dir)

    # Check if the output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    try:
        process_datapoints(source_folder, output_dir)
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
