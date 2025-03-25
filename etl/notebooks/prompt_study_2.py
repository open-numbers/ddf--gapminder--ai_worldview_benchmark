# testing


import polars as pl
import os
import numpy as np
from itertools import combinations
import random
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_evaluation_results():
    """
    Load the evaluation results CSV file.

    Returns:
        pl.DataFrame: A DataFrame containing the evaluation results.
    """
    file_path = os.path.join(
        "../../",
        "ddf--datapoints--evaluation_result--by--question--model_configuration--prompt_variation.csv",
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation results file not found at {file_path}")

    df = pl.read_csv(file_path, schema_overrides={"question": str})

    return df.filter(~pl.col("prompt_variation").str.ends_with("_zh"))


def calculate_correct_rate(df, group_keys=None):
    """
    Calculate the correct rate for evaluation results.

    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        group_keys (list, optional): List of column names to group by.
            If provided, correct rates will be calculated for each group.

    Returns:
        pl.DataFrame: DataFrame with the correct rates.
    """
    # 1. Exclude results with "fail" or "n/a"
    filtered_df = df.filter(~pl.col("evaluation_result").is_in(["fail", "n/a"]))

    if group_keys is None:
        # Calculate the overall correct rate
        total_count = filtered_df.shape[0]
        correct_count = filtered_df.filter(
            pl.col("evaluation_result") == "correct"
        ).shape[0]

        correct_rate = correct_count / total_count if total_count > 0 else 0

        return pl.DataFrame(
            {
                "total_count": [total_count],
                "correct_count": [correct_count],
                "correct_rate": [correct_rate],
            }
        )
    else:
        # Group by the specified keys and calculate correct rate for each group
        return (
            filtered_df.group_by(group_keys)
            .agg(
                [
                    pl.len().alias("total_count"),
                    (pl.col("evaluation_result") == "correct")
                    .sum()
                    .alias("correct_count"),
                ]
            )
            .with_columns(
                [
                    (pl.col("correct_count") / pl.col("total_count")).alias(
                        "correct_rate"
                    )
                ]
            )
        )


def get_baseline_correct_rates(df):
    """
    Calculate baseline correct rates for the full dataset.

    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.

    Returns:
        tuple: (overall_correct_rate, per_question_correct_rates, per_model_correct_rates)
    """
    # Overall correct rate
    overall_rate = calculate_correct_rate(df).get_column("correct_rate")[0]

    # Per-question correct rates
    per_question_rates = calculate_correct_rate(df, group_keys=["question"])

    # Per-model correct rates
    per_model_rates = calculate_correct_rate(df, group_keys=["model_configuration"])

    return overall_rate, per_question_rates, per_model_rates


def calculate_similarity_score(
    baseline_overall,
    baseline_per_question,
    baseline_per_model,
    subset_overall,
    subset_per_question,
    subset_per_model,
    overall_weight=0.3,
    question_weight=0.4,
    model_weight=0.3,
):
    """
    Calculate how similar the subset results are to the baseline results.

    Args:
        baseline_overall (float): Overall correct rate for the full dataset.
        baseline_per_question (pl.DataFrame): Per-question correct rates for the full dataset.
        baseline_per_model (pl.DataFrame): Per-model correct rates for the full dataset.
        subset_overall (float): Overall correct rate for the subset.
        subset_per_question (pl.DataFrame): Per-question correct rates for the subset.
        subset_per_model (pl.DataFrame): Per-model correct rates for the subset.
        overall_weight (float): Weight for the overall rate similarity.
        question_weight (float): Weight for the question-level similarity.
        model_weight (float): Weight for the model-level similarity.

    Returns:
        float: Similarity score (higher is better).
    """
    # Calculate overall rate similarity (lower absolute difference is better)
    overall_similarity = 1 - abs(baseline_overall - subset_overall)

    # Calculate per-question similarity
    # Join the baseline and subset dataframes
    question_comparison = baseline_per_question.join(
        subset_per_question, on="question", how="inner", suffix="_subset"
    )

    # Calculate the mean absolute difference
    question_differences = question_comparison.select(
        (pl.col("correct_rate") - pl.col("correct_rate_subset")).abs().alias("diff")
    )
    mean_question_diff = question_differences.get_column("diff").mean()
    question_similarity = 1 - mean_question_diff

    # Calculate per-model similarity
    model_comparison = baseline_per_model.join(
        subset_per_model, on="model_configuration", how="inner", suffix="_subset"
    )

    model_differences = model_comparison.select(
        (pl.col("correct_rate") - pl.col("correct_rate_subset")).abs().alias("diff")
    )
    mean_model_diff = model_differences.get_column("diff").mean()
    model_similarity = 1 - mean_model_diff

    # Calculate weighted similarity score
    similarity_score = (
        overall_weight * overall_similarity
        + question_weight * question_similarity
        + model_weight * model_similarity
    )

    return similarity_score


def compare_subset_performance(full_df, subset_df):
    """
    Compare the performance of a subset of prompts against the full dataset.

    Args:
        full_df (pl.DataFrame): Full evaluation results DataFrame.
        subset_df (pl.DataFrame): Subset of evaluation results DataFrame.

    Returns:
        dict: Dictionary containing comparison metrics.
    """
    # Calculate per-question correct rates for full and subset
    full_per_question = calculate_correct_rate(full_df, group_keys=["question", "model_configuration"])
    subset_per_question = calculate_correct_rate(subset_df, group_keys=["question", "model_configuration"])

    # Join datasets to compare
    comparison = full_per_question.join(
        subset_per_question, on=["question", "model_configuration"], how="inner", suffix="_subset"
    )

    # Calculate differences
    comparison = comparison.with_columns(
        [(pl.col("correct_rate") - pl.col("correct_rate_subset")).abs().alias("diff")]
    )

    # Get max difference and average difference
    max_diff = comparison.get_column("diff").max()
    avg_diff = comparison.get_column("diff").mean()

    # Get the question with the maximum difference
    max_diff_row = (
        comparison.filter(pl.col("diff") == max_diff)
        .select(["question", "correct_rate", "correct_rate_subset", "diff"])
        .to_dicts()[0]
    )

    return {
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "max_diff_question": max_diff_row,
    }


def evaluate_prompt_subset(df, prompt_subset):
    """
    Evaluate how well a subset of prompts represents the full test.

    Args:
        df (pl.DataFrame): Full evaluation results DataFrame.
        prompt_subset (list): List of prompt_variation IDs to evaluate.

    Returns:
        float: Similarity score (higher is better).
    """
    # Calculate baseline rates with the full dataset
    baseline_overall, baseline_per_question, baseline_per_model = (
        get_baseline_correct_rates(df)
    )

    # Filter to only include the selected prompts
    subset_df = df.filter(pl.col("prompt_variation").is_in(prompt_subset))

    # Calculate rates for the subset
    subset_overall = calculate_correct_rate(subset_df).get_column("correct_rate")[0]
    subset_per_question = calculate_correct_rate(subset_df, group_keys=["question"])
    subset_per_model = calculate_correct_rate(
        subset_df, group_keys=["model_configuration"]
    )

    # Calculate similarity score
    similarity = calculate_similarity_score(
        baseline_overall,
        baseline_per_question,
        baseline_per_model,
        subset_overall,
        subset_per_question,
        subset_per_model,
    )

    return similarity


def find_representative_prompts(
    df, num_prompts=15, num_iterations=100, method="greedy"
):
    """
    Find a set of representative prompt variations.

    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        num_prompts (int): Number of prompt variations to select.
        num_iterations (int): Number of iterations to run for optimization.
        method (str): Method to use for selection ("random", "greedy", "heuristic", or "best_worst_random").

    Returns:
        tuple: (best_prompt_set, similarity_score)
    """
    # Get all unique prompt variations
    all_prompts = df.get_column("prompt_variation").unique()

    # Calculate baseline rates
    baseline_overall, baseline_per_question, baseline_per_model = (
        get_baseline_correct_rates(df)
    )

    best_score = 0
    best_prompt_set = []

    if method == "random":
        # Random sampling approach
        for _ in range(num_iterations):
            prompt_set = random.sample(all_prompts.to_list(), num_prompts)
            score = evaluate_prompt_subset(df, prompt_set)

            if score > best_score:
                best_score = score
                best_prompt_set = prompt_set

    elif method == "greedy":
        # Greedy approach: add the best prompt one at a time
        current_set = []
        remaining_prompts = all_prompts.to_list()

        for _ in range(num_prompts):
            best_prompt = None
            best_iteration_score = 0

            for prompt in remaining_prompts:
                candidate_set = current_set + [prompt]
                score = evaluate_prompt_subset(df, candidate_set)

                if score > best_iteration_score:
                    best_iteration_score = score
                    best_prompt = prompt

            if best_prompt:
                current_set.append(best_prompt)
                remaining_prompts.remove(best_prompt)

        best_prompt_set = current_set
        best_score = evaluate_prompt_subset(df, best_prompt_set)

    elif method == "heuristic":
        # Heuristic approach: select prompts based on their individual performance
        # Calculate correct rate for each prompt
        prompt_rates = calculate_correct_rate(df, group_keys=["prompt_variation"])

        # Sort by how close they are to the overall baseline
        prompt_rates = prompt_rates.with_columns(
            [
                (pl.col("correct_rate") - baseline_overall)
                .abs()
                .alias("diff_from_baseline")
            ]
        )

        # Take the top num_prompts/2 closest to baseline
        closest_prompts = (
            prompt_rates.sort("diff_from_baseline")
            .head(num_prompts // 2)
            .get_column("prompt_variation")
            .to_list()
        )

        # Take some prompts with higher and lower correct rates to ensure diversity
        higher_prompts = (
            prompt_rates.filter(pl.col("correct_rate") > baseline_overall)
            .sort("correct_rate", descending=True)
            .head(num_prompts // 4)
            .get_column("prompt_variation")
            .to_list()
        )
        lower_prompts = (
            prompt_rates.filter(pl.col("correct_rate") < baseline_overall)
            .sort("correct_rate")
            .head(num_prompts // 4)
            .get_column("prompt_variation")
            .to_list()
        )

        # Combine the sets and take the first num_prompts
        candidate_set = closest_prompts + higher_prompts + lower_prompts
        # Remove any duplicates
        candidate_set = list(dict.fromkeys(candidate_set))
        # Ensure we have exactly num_prompts
        if len(candidate_set) > num_prompts:
            candidate_set = candidate_set[:num_prompts]
        elif len(candidate_set) < num_prompts:
            # Add more prompts if needed
            remaining = [p for p in all_prompts.to_list() if p not in candidate_set]
            candidate_set += random.sample(remaining, num_prompts - len(candidate_set))

        best_prompt_set = candidate_set
        best_score = evaluate_prompt_subset(df, best_prompt_set)

    elif method == "best_worst_random":
        # First calculate correct rate for each prompt
        prompt_rates = calculate_correct_rate(df, group_keys=["prompt_variation"])

        # Sort prompts by correct rate
        sorted_prompts = prompt_rates.sort("correct_rate")

        # Get the best and worst performing prompts
        best_count = 2
        worst_count = 2

        # Select best prompts (highest correct rate)
        best_prompts = (
            sorted_prompts.tail(best_count).get_column("prompt_variation").to_list()
        )

        # Select worst prompts (lowest correct rate)
        worst_prompts = (
            sorted_prompts.head(worst_count).get_column("prompt_variation").to_list()
        )

        # Combine best and worst
        candidate_set = best_prompts + worst_prompts
        for _ in range(num_iterations):
            # Fill remaining slots with random sampling
            remaining_slots = num_prompts - len(candidate_set)
            if remaining_slots > 0:
                remaining_prompts = [
                    p for p in all_prompts.to_list() if p not in candidate_set
                ]
                # Use different random samples for each iteration
                random_prompts = random.sample(remaining_prompts, remaining_slots)
                candidate_set += random_prompts

            # Evaluate this candidate set
            score = evaluate_prompt_subset(df, candidate_set)

            # Update best set if this one is better
            if score > best_score:
                best_score = score
                best_prompt_set = candidate_set

    return best_prompt_set, best_score


def find_optimal_prompt_set(df, num_prompts=15):
    """
    Find the optimal set of prompts by trying different methods and iterations.

    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        num_prompts (int): Number of prompt variations to select.

    Returns:
        dict: Results containing the best prompt set and details about its performance.
    """
    # Add the new best_worst_random method to the list
    methods = ["random", "heuristic", "best_worst_random"]
    iterations = [200, 1, 200]  # Run 100 iterations for random and best_worst_random

    best_score = 0
    best_method = None
    best_prompt_set = []

    for method, num_iterations in zip(methods, iterations):
        print(f"Trying {method} method with {num_iterations} iterations...")
        prompt_set, score = find_representative_prompts(
            df, num_prompts, num_iterations, method
        )
        print(f"{method} method score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_method = method
            best_prompt_set = prompt_set

    # Calculate detailed metrics for the best set
    subset_df = df.filter(pl.col("prompt_variation").is_in(best_prompt_set))

    # Get baseline and subset metrics
    baseline_overall, baseline_per_question, baseline_per_model = (
        get_baseline_correct_rates(df)
    )
    subset_overall = calculate_correct_rate(subset_df).get_column("correct_rate")[0]
    subset_per_question = calculate_correct_rate(subset_df, group_keys=["question"])
    subset_per_model = calculate_correct_rate(
        subset_df, group_keys=["model_configuration"]
    )

    # Compare subset performance
    performance_comparison = compare_subset_performance(df, subset_df)

    return {
        "best_prompt_set": best_prompt_set,
        "similarity_score": best_score,
        "method": best_method,
        "baseline_overall_rate": baseline_overall,
        "subset_overall_rate": subset_overall,
        "baseline_per_question": baseline_per_question,
        "subset_per_question": subset_per_question,
        "baseline_per_model": baseline_per_model,
        "subset_per_model": subset_per_model,
        "max_question_diff": performance_comparison["max_diff"],
        "avg_question_diff": performance_comparison["avg_diff"],
        "max_diff_question": performance_comparison["max_diff_question"],
    }


def compare_model_performance(full_df, subset_df, save_path=None):
    """
    Compare model performance between the full dataset and a subset.
    
    Args:
        full_df (pl.DataFrame): Full evaluation results DataFrame.
        subset_df (pl.DataFrame): Subset of evaluation results DataFrame.
        save_path (str, optional): Path to save the visualization. If None, the 
                                   plot will be shown but not saved.
                                  
    Returns:
        tuple: (comparison_data, figure)
            - comparison_data (pl.DataFrame): DataFrame with model performance comparisons
            - figure (matplotlib.figure.Figure): Figure object with the visualization
    """
    # Calculate per-model correct rates for full and subset
    full_per_model = calculate_correct_rate(full_df, group_keys=["model_configuration"])
    subset_per_model = calculate_correct_rate(subset_df, group_keys=["model_configuration"])
    
    # Join the dataframes to compare
    comparison = full_per_model.join(
        subset_per_model, on="model_configuration", how="inner", suffix="_subset"
    )
    
    # Calculate absolute differences and relative differences
    comparison = comparison.with_columns([
        (pl.col("correct_rate") - pl.col("correct_rate_subset")).abs().alias("abs_diff"),
        ((pl.col("correct_rate") - pl.col("correct_rate_subset")) / pl.col("correct_rate") * 100).alias("rel_diff_percent")
    ])
    
    # Sort by model name for better visualization
    comparison = comparison.sort("model_configuration")
    
    # Create a visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Get model names and performance values
    models = comparison.get_column("model_configuration").to_list()
    full_rates = comparison.get_column("correct_rate").to_list()
    subset_rates = comparison.get_column("correct_rate_subset").to_list()
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35
    
    # Plot performance bars
    bar1 = ax1.bar(x - width/2, full_rates, width, label='Full Dataset', color='royalblue')
    bar2 = ax1.bar(x + width/2, subset_rates, width, label='Subset', color='lightcoral')
    
    # Add labels and title
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Correct Rate')
    ax1.set_title('Model Performance Comparison: Full Dataset vs. Subset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    
    # Add text annotations showing the exact values
    # for i, (v1, v2) in enumerate(zip(full_rates, subset_rates)):
    #     ax1.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
    #     ax1.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot difference
    abs_diffs = comparison.get_column("abs_diff").to_list()
    colors = ['red' if diff > 0.05 else 'orange' if diff > 0.02 else 'green' for diff in abs_diffs]
    ax2.bar(x, abs_diffs, width, color=colors)
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Absolute Difference in Correct Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    # Add a horizontal line for the average difference
    avg_diff = np.mean(abs_diffs)
    ax2.axhline(y=avg_diff, color='black', linestyle='--', alpha=0.7, 
                label=f'Avg Diff: {avg_diff:.3f}')
    ax2.legend()
    
    # Add text annotations showing the exact differences
    for i, diff in enumerate(abs_diffs):
        ax2.text(i, diff + 0.002, f'{diff:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model performance comparison saved to {save_path}")
    
    return comparison, fig


# Usage example
if __name__ == "__main__":
    df = load_evaluation_results()

    # Calculate correct rates
    print("Calculating overall correct rate...")
    overall_rate = calculate_correct_rate(df)
    print(f"Overall correct rate: {overall_rate.get_column('correct_rate')[0]:.4f}")

    # Find representative prompts
    print("\nFinding 15 representative prompts...")
    results = find_optimal_prompt_set(df, num_prompts=15)

    print("\nResults:")
    print(f"Best method: {results['method']}")
    print(f"Similarity score: {results['similarity_score']:.4f}")
    print(f"Baseline overall rate: {results['baseline_overall_rate']:.4f}")
    print(f"Subset overall rate: {results['subset_overall_rate']:.4f}")
    print(f"Average question difference: {results['avg_question_diff']:.4f}")
    print(f"Maximum question difference: {results['max_question_diff']:.4f}")
    print(f"Question with max difference: {results['max_diff_question']}")
    print(f"Best prompt set: {results['best_prompt_set']}")

    # Compare model performance
    print("\nComparing model performance...")
    subset_df = df.filter(pl.col("prompt_variation").is_in(results["best_prompt_set"]))
    model_comparison, fig = compare_model_performance(
        df, subset_df, save_path="model_performance_comparison.png"
    )
    
    # Display summary stats about model differences
    abs_diffs = model_comparison.get_column("abs_diff")
    max_diff = abs_diffs.max()
    avg_diff = abs_diffs.mean()
    
    max_diff_model = model_comparison.filter(pl.col("abs_diff") == max_diff).get_column("model_configuration")[0]
    
    print(f"\nModel Performance Comparison:")
    print(f"Average absolute difference in correct rate: {avg_diff:.4f}")
    print(f"Maximum absolute difference in correct rate: {max_diff:.4f} (Model: {max_diff_model})")
    
    # Output the optimal set of prompt variations
    print("\nOptimal set of 15 prompt variations:")
    for i, prompt_id in enumerate(results["best_prompt_set"], 1):
        print(f"{i}. {prompt_id}")
