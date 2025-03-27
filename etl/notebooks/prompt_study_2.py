# # Select representative prompts
# In this notebook, I try a few different methods to select a part of prompt variations so that they can represent entire prompt set.  


import polars as pl
import os
import numpy as np
from itertools import combinations
import random
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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

    # do not include the chinese prompts, 
    # and also the "source_wikipedia" prompt because it's new and don't have enough datapoints.
    # return df.filter(~pl.col("prompt_variation").str.ends_with("_zh"), pl.col("prompt_variation") != "source_wikipedia")
    return df.filter(~pl.col("prompt_variation").str.ends_with("_zh"))


def add_prompt_category(df):
    """
    Extract the prefix/category from prompt variation IDs and add it as a new column.
    
    Args:
        df (pl.DataFrame): DataFrame containing a 'prompt_variation' column
        
    Returns:
        pl.DataFrame: DataFrame with an added 'prompt_category' column
    """
    # Extract the category from prompt variation IDs
    # Assuming prompt variations have format like "category_specific_details"
    # e.g., "zero_shot", "few_shot_3", "cot_reasoning", etc.
    return df.with_columns([
        pl.col("prompt_variation")
            .str.split("_")
            .list.get(0)
            .alias("prompt_category")
    ])


def calculate_correct_rate(df, group_keys=None, exclude_fail=False):
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
    if exclude_fail:
        filtered_df = df.filter(~pl.col("evaluation_result").is_in(["fail", "n/a"]))
    else:
        filtered_df = df

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
        tuple: (overall_correct_rate, per_question_model_correct_rates, per_model_correct_rates)
    """
    # Overall correct rate
    overall_rate = calculate_correct_rate(df).get_column("correct_rate")[0]

    # Per-question and per-model correct rates
    per_question_model_rates = calculate_correct_rate(df, group_keys=["question", "model_configuration"])

    # Per-model correct rates
    per_model_rates = calculate_correct_rate(df, group_keys=["model_configuration"])

    return overall_rate, per_question_model_rates, per_model_rates



def pca_based_selection(df, num_prompts=15):
    """
    Select representative prompts using PCA-based dimensionality reduction.
    
    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        num_prompts (int): Number of prompt variations to select.
        
    Returns:
        list: Selected prompt variations.
    """
    # Get unique prompts, questions, and models
    prompts = df.get_column("prompt_variation").unique().to_list()
    
    print(f"Building performance matrix with {len(prompts)} prompts...")
    
    # Pre-calculate all correct rates in a single operation
    # This creates a dataframe with correct_rate for each prompt-question-model combination
    all_rates = calculate_correct_rate(
        df, 
        group_keys=["prompt_variation", "question", "model_configuration"]
    )
    
    # Get all unique question-model pairs for constructing the performance matrix
    all_qm_pairs = df.select(["question", "model_configuration"]).unique()
    print(f"Found {len(all_qm_pairs)} question-model pairs")
    
    # Create the performance matrix - each row is a prompt, each column is a question-model pair
    performance_matrix = []
    
    for prompt in prompts:
        # Filter rates for this prompt
        prompt_rates = all_rates.filter(pl.col("prompt_variation") == prompt)
        prompt_vector = []
        
        for qm_row in all_qm_pairs.iter_rows(named=True):
            question, model = qm_row["question"], qm_row["model_configuration"]
            
            # Find the correct rate for this prompt-question-model combination
            matching_rate = prompt_rates.filter(
                (pl.col("question") == question) & 
                (pl.col("model_configuration") == model)
            )
            
            if matching_rate.shape[0] > 0:
                # Use the pre-calculated correct rate
                correct_rate = matching_rate.get_column("correct_rate")[0]
                prompt_vector.append(correct_rate)
            else:
                # Handle missing data
                prompt_vector.append(0)
                
        performance_matrix.append(prompt_vector)
    
    # Convert to numpy array and standardize
    X = np.array(performance_matrix)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply PCA
    n_components = min(num_prompts, len(prompts)-1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variance explained by {n_components} components: {explained_variance:.4f}")
    
    # Calculate prompt importance scores based on their representation in PCA space
    importance_scores = {}
    for i, prompt in enumerate(prompts):
        # Calculate Euclidean distance in PCA space (considering all components)
        importance = np.linalg.norm(X_pca[i])
        importance_scores[prompt] = importance
    
    # Use K-means clustering in PCA space to select diverse prompts
    kmeans = KMeans(n_clusters=num_prompts, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    # Select the prompt closest to each cluster center
    selected_prompts = []
    for i in range(num_prompts):
        cluster_points = np.array([X_pca[j] for j, c in enumerate(clusters) if c == i])
        cluster_original_indices = [j for j, c in enumerate(clusters) if c == i]
        
        if len(cluster_points) > 0:
            # Find prompt closest to cluster center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_point_idx = cluster_original_indices[np.argmin(distances)]
            selected_prompts.append(prompts[closest_point_idx])
    
    # Print top prompts by importance for informational purposes
    print("\nTop prompts by PCA importance score:")
    sorted_prompts = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    for prompt, score in sorted_prompts[:35]:
        print(f"{prompt}: {score:.4f}")
    
    return selected_prompts


def kmeans_based_selection(df, num_prompts=15):
    """
    Select representative prompts using K-means clustering directly on the performance matrix.
    
    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        num_prompts (int): Number of prompt variations to select.
        
    Returns:
        list: Selected prompt variations.
    """
    # Get unique prompts, questions, and models
    prompts = df.get_column("prompt_variation").unique().to_list()
    
    print(f"Building performance matrix with {len(prompts)} prompts...")
    
    # Pre-calculate all correct rates in a single operation
    all_rates = calculate_correct_rate(
        df, 
        group_keys=["prompt_variation", "question", "model_configuration"]
    )
    
    # Get all unique question-model pairs
    all_qm_pairs = df.select(["question", "model_configuration"]).unique()
    print(f"Found {len(all_qm_pairs)} question-model pairs")
    
    # Create the performance matrix - each row is a prompt, each column is a question-model pair
    performance_matrix = []
    
    for prompt in prompts:
        # Filter rates for this prompt
        prompt_rates = all_rates.filter(pl.col("prompt_variation") == prompt)
        prompt_vector = []
        
        for qm_row in all_qm_pairs.iter_rows(named=True):
            question, model = qm_row["question"], qm_row["model_configuration"]
            
            # Find the correct rate for this prompt-question-model combination
            matching_rate = prompt_rates.filter(
                (pl.col("question") == question) & 
                (pl.col("model_configuration") == model)
            )
            
            if matching_rate.shape[0] > 0:
                correct_rate = matching_rate.get_column("correct_rate")[0]
                prompt_vector.append(correct_rate)
            else:
                prompt_vector.append(0)  # Handle missing data
                
        performance_matrix.append(prompt_vector)
    
    # Convert to numpy array and standardize
    X = np.array(performance_matrix)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply K-means clustering directly on the scaled data
    print(f"Applying K-means clustering with {num_prompts} clusters...")
    kmeans = KMeans(n_clusters=num_prompts, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Select the prompt closest to each cluster center
    selected_prompts = []
    for i in range(num_prompts):
        cluster_points = np.array([X_scaled[j] for j, c in enumerate(clusters) if c == i])
        cluster_original_indices = [j for j, c in enumerate(clusters) if c == i]
        
        if len(cluster_points) > 0:
            # Find prompt closest to cluster center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_point_idx = cluster_original_indices[np.argmin(distances)]
            selected_prompts.append(prompts[closest_point_idx])
    
    return selected_prompts


def calculate_similarity_score(
    baseline_overall,
    baseline_per_question_model,
    baseline_per_model,
    subset_overall,
    subset_per_question_model,
    subset_per_model,
    overall_weight=0.1,
    question_model_weight=0.8,
    model_weight=0.1,
):
    """
    Calculate how similar the subset results are to the baseline results.

    Args:
        baseline_overall (float): Overall correct rate for the full dataset.
        baseline_per_question_model (pl.DataFrame): Per-question and per-model correct rates for the full dataset.
        baseline_per_model (pl.DataFrame): Per-model correct rates for the full dataset.
        subset_overall (float): Overall correct rate for the subset.
        subset_per_question_model (pl.DataFrame): Per-question and per-model correct rates for the subset.
        subset_per_model (pl.DataFrame): Per-model correct rates for the subset.
        overall_weight (float): Weight for the overall rate similarity.
        question_model_weight (float): Weight for the question-and-model-level similarity.
        model_weight (float): Weight for the model-level similarity.

    Returns:
        float: Similarity score (higher is better).
    """
    # Calculate overall rate similarity (lower absolute difference is better)
    overall_similarity = 1 - abs(baseline_overall - subset_overall)

    # Calculate per-question-model similarity
    # Join the baseline and subset dataframes
    question_model_comparison = baseline_per_question_model.join(
        subset_per_question_model, on=["question", "model_configuration"], how="inner", suffix="_subset"
    )

    # Calculate the mean absolute difference
    question_model_differences = question_model_comparison.select(
        (pl.col("correct_rate") - pl.col("correct_rate_subset")).abs().alias("diff")
    )
    mean_question_model_diff = question_model_differences.get_column("diff").mean()
    question_model_similarity = 1 - mean_question_model_diff

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
        + question_model_weight * question_model_similarity
        + model_weight * model_similarity
    )

    return similarity_score


def get_top_bottom_prompt_rates(df, top_n=5, bottom_n=5):
    """
    Get the average correct rates for the top N and bottom N performing prompts per model.
    
    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        top_n (int): Number of top performing prompts to include per model.
        bottom_n (int): Number of bottom performing prompts to include per model.
        
    Returns:
        dict: Dictionary containing comparison metrics
            - per_model_top_rates: Dict mapping model to top prompts average rate
            - per_model_bottom_rates: Dict mapping model to bottom prompts average rate
            - avg_top_rate: Average of top prompt rates across all models
            - avg_bottom_rate: Average of bottom prompt rates across all models
            - model_top_diffs: List of differences in top prompt rates per model
            - model_bottom_diffs: List of differences in bottom prompt rates per model
    """
    # Get all unique models
    models = df.get_column("model_configuration").unique().to_list()
    
    # Initialize dictionaries to store results
    full_per_model_top_rates = {}
    full_per_model_bottom_rates = {}
    full_per_model_top_prompts = {}
    full_per_model_bottom_prompts = {}
    
    # Process each model separately
    for model in models:
        # Filter data for this model
        model_df = df.filter(pl.col("model_configuration") == model)
        
        # Calculate correct rate for each prompt variation for this model
        prompt_rates = calculate_correct_rate(model_df, group_keys=["prompt_variation"], exclude_fail=True)
        
        # Skip if there aren't enough unique prompts for this model
        if prompt_rates.shape[0] < top_n + bottom_n:
            continue
        
        # Sort prompts by correct rate
        sorted_prompts = prompt_rates.sort("correct_rate")
        
        # Get bottom N prompts and their rates
        bottom_prompts = sorted_prompts.head(bottom_n)
        bottom_prompts_avg_rate = bottom_prompts.get_column("correct_rate").mean()
        bottom_prompts_list = bottom_prompts.select(["prompt_variation", "correct_rate"])
        
        # Get top N prompts and their rates
        top_prompts = sorted_prompts.tail(top_n)
        top_prompts_avg_rate = top_prompts.get_column("correct_rate").mean()
        top_prompts_list = top_prompts.select(["prompt_variation", "correct_rate"])
        
        # Store the rates for this model
        full_per_model_top_rates[model] = top_prompts_avg_rate
        full_per_model_bottom_rates[model] = bottom_prompts_avg_rate
        full_per_model_top_prompts[model] = top_prompts_list
        full_per_model_bottom_prompts[model] = bottom_prompts_list
    
    # Calculate the average rates across all models
    avg_top_rate = sum(full_per_model_top_rates.values()) / len(full_per_model_top_rates) if full_per_model_top_rates else 0
    avg_bottom_rate = sum(full_per_model_bottom_rates.values()) / len(full_per_model_bottom_rates) if full_per_model_bottom_rates else 0
    
    return {
        "per_model_top_rates": full_per_model_top_rates,
        "per_model_bottom_rates": full_per_model_bottom_rates,
        "avg_top_rate": avg_top_rate,
        "avg_bottom_rate": avg_bottom_rate,
        "per_model_top_prompts": full_per_model_top_prompts,
        "per_model_bottom_prompts": full_per_model_bottom_prompts
    }


def compare_top_bottom_prompt_rates(full_df, subset_df, top_n=5, bottom_n=5):
    """
    Compare the top/bottom prompt rates between full dataset and subset per model.
    
    Args:
        full_df (pl.DataFrame): Full evaluation results DataFrame.
        subset_df (pl.DataFrame): Subset of evaluation results DataFrame.
        top_n (int): Number of top performing prompts to include per model.
        bottom_n (int): Number of bottom performing prompts to include per model.
        
    Returns:
        dict: Dictionary containing comparison metrics
    """
    # Get rates for full dataset
    full_rates = get_top_bottom_prompt_rates(full_df, top_n, bottom_n)
    
    # Get rates for subset
    subset_rates = get_top_bottom_prompt_rates(subset_df, top_n, bottom_n)
    
    # Get common models between full and subset
    common_models = set(full_rates["per_model_top_rates"].keys()).intersection(
        set(subset_rates["per_model_top_rates"].keys()))
    
    # Calculate per-model differences
    per_model_top_diffs = {}
    per_model_bottom_diffs = {}
    
    for model in common_models:
        top_diff = abs(full_rates["per_model_top_rates"][model] - 
                      subset_rates["per_model_top_rates"][model])
        bottom_diff = abs(full_rates["per_model_bottom_rates"][model] - 
                         subset_rates["per_model_bottom_rates"][model])
        
        per_model_top_diffs[model] = top_diff
        per_model_bottom_diffs[model] = bottom_diff
    
    # Calculate average differences
    top_diff_values = list(per_model_top_diffs.values())
    bottom_diff_values = list(per_model_bottom_diffs.values())
    
    avg_top_diff = sum(top_diff_values) / len(top_diff_values) if top_diff_values else 0
    avg_bottom_diff = sum(bottom_diff_values) / len(bottom_diff_values) if bottom_diff_values else 0
    
    # Calculate overall average metrics
    full_avg_top = full_rates["avg_top_rate"]
    full_avg_bottom = full_rates["avg_bottom_rate"]
    subset_avg_top = subset_rates["avg_top_rate"]
    subset_avg_bottom = subset_rates["avg_bottom_rate"]
    
    return {
        "full_per_model_top_rates": full_rates["per_model_top_rates"],
        "full_per_model_bottom_rates": full_rates["per_model_bottom_rates"],
        "subset_per_model_top_rates": subset_rates["per_model_top_rates"],
        "subset_per_model_bottom_rates": subset_rates["per_model_bottom_rates"],
        "per_model_top_diffs": per_model_top_diffs,
        "per_model_bottom_diffs": per_model_bottom_diffs,
        "top_diff_values": top_diff_values,
        "bottom_diff_values": bottom_diff_values,
        "avg_top_diff": avg_top_diff,
        "avg_bottom_diff": avg_bottom_diff,
        "full_avg_top": full_avg_top,
        "full_avg_bottom": full_avg_bottom,
        "subset_avg_top": subset_avg_top,
        "subset_avg_bottom": subset_avg_bottom,
        "full_per_model_top_prompts": full_rates["per_model_top_prompts"],
        "full_per_model_bottom_prompts": full_rates["per_model_bottom_prompts"]
    }


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
    full_per_question = calculate_correct_rate(full_df, group_keys=["question", "model_configuration"], exclude_fail=True)
    subset_per_question = calculate_correct_rate(subset_df, group_keys=["question", "model_configuration"], exclude_fail=True)

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
    
    # Calculate additional statistics
    median_diff = comparison.get_column("diff").median()
    percentile_25 = comparison.get_column("diff").quantile(0.25)
    percentile_75 = comparison.get_column("diff").quantile(0.75)

    # Get the question with the maximum difference
    max_diff_row = (
        comparison.filter(pl.col("diff") == max_diff)
        .select(["question", "model_configuration", "correct_rate", "correct_rate_subset", "diff"])
        .to_dicts()[0]
    )
    
    # Compare top and bottom prompt rates per model
    prompt_comparison = compare_top_bottom_prompt_rates(full_df, subset_df)

    return {
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "median_diff": median_diff,
        "percentile_25": percentile_25,
        "percentile_75": percentile_75,
        "max_diff_question": max_diff_row,
        "diff_distribution": comparison.get_column("diff").to_list(),
        "question_model_comparison": comparison,
        
        # Top/bottom prompt metrics per model
        "full_per_model_top_rates": prompt_comparison["full_per_model_top_rates"],
        "full_per_model_bottom_rates": prompt_comparison["full_per_model_bottom_rates"],
        "subset_per_model_top_rates": prompt_comparison["subset_per_model_top_rates"],
        "subset_per_model_bottom_rates": prompt_comparison["subset_per_model_bottom_rates"],
        "per_model_top_diffs": prompt_comparison["per_model_top_diffs"],
        "per_model_bottom_diffs": prompt_comparison["per_model_bottom_diffs"],
        
        # Average metrics across all models
        "full_avg_top": prompt_comparison["full_avg_top"],
        "full_avg_bottom": prompt_comparison["full_avg_bottom"],
        "subset_avg_top": prompt_comparison["subset_avg_top"],
        "subset_avg_bottom": prompt_comparison["subset_avg_bottom"],
        "avg_top_diff": prompt_comparison["avg_top_diff"],
        "avg_bottom_diff": prompt_comparison["avg_bottom_diff"],
        
        # Detailed prompt info
        "full_per_model_top_prompts": prompt_comparison["full_per_model_top_prompts"],
        "full_per_model_bottom_prompts": prompt_comparison["full_per_model_bottom_prompts"]
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
    baseline_overall, baseline_per_question_model, baseline_per_model = (
        get_baseline_correct_rates(df)
    )

    # Filter to only include the selected prompts
    subset_df = df.filter(pl.col("prompt_variation").is_in(prompt_subset))

    # Calculate rates for the subset
    subset_overall = calculate_correct_rate(subset_df).get_column("correct_rate")[0]
    subset_per_question_model = calculate_correct_rate(subset_df, group_keys=["question", "model_configuration"])
    subset_per_model = calculate_correct_rate(
        subset_df, group_keys=["model_configuration"]
    )

    # Calculate similarity score
    similarity = calculate_similarity_score(
        baseline_overall,
        baseline_per_question_model,
        baseline_per_model,
        subset_overall,
        subset_per_question_model,
        subset_per_model,
    )

    return similarity


def calculate_prompt_disagreement(df):
    """
    Calculate how much disagreement/variance each prompt creates across models.

    For each question and prompt combination, we measure how often models give
    different answers, then aggregate to find which prompts create the most
    disagreement among models.

    Args:
        df (pl.DataFrame): DataFrame containing evaluation results.

    Returns:
        pl.DataFrame: DataFrame with disagreement scores for each prompt variation.
    """
    # Group by question and prompt_variation
    disagreement_by_qp = (
        df.group_by(["question", "prompt_variation"])
        .agg([
            pl.len().alias("total_responses"),
            pl.col("model_configuration").n_unique().alias("total_models"),
            pl.col("evaluation_result").n_unique().alias("unique_answers"),
            # Count each type of result for more detailed analysis
            (pl.col("evaluation_result") == "correct").sum().alias("correct_count"),
            (pl.col("evaluation_result") == "wrong").sum().alias("wrong_count"),
            (pl.col("evaluation_result") == "very_wrong").sum().alias("very_wrong_count"),
            (pl.col("evaluation_result").is_in(["fail", "n/a"])).sum().alias("indecisive_count")
        ])
        .with_columns([
            # Calculate disagreement ratio (0 = all agree, 1 = all disagree)
            ((pl.col("unique_answers") - 1) / (pl.col("total_models").clip(2, None) - 1)).alias("disagreement_ratio"),

            # Calculate most common result proportion
            (pl.max_horizontal(
                pl.col("correct_count"),
                pl.col("wrong_count"),
                pl.col("very_wrong_count"),
                pl.col("indecisive_count")
            ) / pl.col("total_responses")).alias("dominant_answer_ratio")
        ])
    )

    # Aggregate by prompt_variation
    prompt_disagreement = (
        disagreement_by_qp.group_by("prompt_variation")
        .agg([
            # Average disagreement across all questions
            pl.col("disagreement_ratio").mean().alias("avg_disagreement"),
            # Alternative measure: 1 - average dominant answer ratio
            (1 - pl.col("dominant_answer_ratio").mean()).alias("avg_diversity"),
            # Number of questions
            pl.len().alias("question_count")
        ])
        .sort("avg_disagreement", descending=True)
    )

    return prompt_disagreement


def analyze_prompt_disagreement(df):
    """
    Analyze prompt disagreement and print insights.

    Args:
        df (pl.DataFrame): DataFrame containing evaluation results.

    Returns:
        pl.DataFrame: DataFrame with disagreement analysis.
    """
    # Calculate disagreement metrics
    disagreement_df = calculate_prompt_disagreement(df)

    # Add prompt category for additional analysis
    disagreement_df = add_prompt_category(disagreement_df)

    # Calculate correct rate per prompt for correlation analysis
    correct_rates = calculate_correct_rate(df, group_keys=["prompt_variation"])

    # Join correct rates with disagreement data
    analysis_df = disagreement_df.join(
        correct_rates.select(["prompt_variation", "correct_rate"]),
        on="prompt_variation",
        how="left"
    )

    # Print summary statistics
    print("=== Prompt Disagreement Analysis ===")
    print(f"Average disagreement across all prompts: {analysis_df['avg_disagreement'].mean():.4f}")
    print(f"Most divisive prompt: {analysis_df.sort('avg_disagreement', descending=True).head(1)['prompt_variation'][0]}")
    print(f"Least divisive prompt: {analysis_df.sort('avg_disagreement').head(1)['prompt_variation'][0]}")

    # Analyze by prompt category if available
    if "prompt_category" in analysis_df.columns:
        category_analysis = (
            analysis_df.group_by("prompt_category")
            .agg([
                pl.col("avg_disagreement").mean().alias("avg_category_disagreement"),
                pl.col("correct_rate").mean().alias("avg_category_correct_rate"),
                pl.len().alias("prompt_count")
            ])
            .sort("avg_category_disagreement", descending=True)
        )

        print("\n=== Disagreement by Prompt Category ===")
        for row in category_analysis.iter_rows(named=True):
            print(f"{row['prompt_category']}: {row['avg_category_disagreement']:.4f} disagreement, "
                  f"{row['avg_category_correct_rate']:.4f} correct rate ({row['prompt_count']} prompts)")

    # Find top 5 most and least divisive prompts
    print("\n=== Top 5 Most Divisive Prompts ===")
    most_divisive = analysis_df.sort("avg_disagreement", descending=True).head(5)
    for row in most_divisive.iter_rows(named=True):
        print(f"{row['prompt_variation']}: {row['avg_disagreement']:.4f} disagreement, {row['correct_rate']:.4f} correct rate")

    print("\n=== Top 5 Least Divisive Prompts ===")
    least_divisive = analysis_df.sort("avg_disagreement").head(5)
    for row in least_divisive.iter_rows(named=True):
        print(f"{row['prompt_variation']}: {row['avg_disagreement']:.4f} disagreement, {row['correct_rate']:.4f} correct rate")

    return analysis_df


def generate_disagreement_report(analysis_df, output_dir="disagreement_analysis"):
    """
    Generate CSV reports from the disagreement analysis.

    Args:
        analysis_df (pl.DataFrame): DataFrame with disagreement analysis.
        output_dir (str): Directory to save CSV files to.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save main analysis dataframe
    main_path = os.path.join(output_dir, "prompt_disagreement_analysis.csv")
    analysis_df.write_csv(main_path)
    print(f"Saved prompt disagreement analysis to {main_path}")

    # Save category summary if available
    if "prompt_category" in analysis_df.columns:
        category_analysis = (
            analysis_df.group_by("prompt_category")
            .agg([
                pl.col("avg_disagreement").mean().alias("avg_disagreement"),
                pl.col("avg_diversity").mean().alias("avg_diversity"),
                pl.col("correct_rate").mean().alias("avg_correct_rate"),
                pl.len().alias("prompt_count")
            ])
            .sort("avg_disagreement", descending=True)
        )

        category_path = os.path.join(output_dir, "disagreement_by_category.csv")
        category_analysis.write_csv(category_path)
        print(f"Saved category disagreement analysis to {category_path}")


def alternating_category_selection(df, num_prompts=15, metric="disagreement"):
    """
    Select representative prompts by taking top and bottom prompts from each category,
    then filling remaining slots with overall top and bottom prompts.

    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        num_prompts (int): Number of prompt variations to select.
        metric (str): Which metric to use - "disagreement" or "diversity"

    Returns:
        list: Selected prompt variations.
    """
    # do it my own
    # Get disagreement analysis
    disagreement_df = calculate_prompt_disagreement(df)
    disagreement_df = add_prompt_category(disagreement_df)

    # Join with correct rates
    correct_rates = calculate_correct_rate(df, group_keys=["prompt_variation"])
    analysis_df = disagreement_df.join(
        correct_rates.select(["prompt_variation", "correct_rate"]),
        on="prompt_variation",
        how="left"
    )

    # Choose which metric to sort by
    if metric == "disagreement":
        sort_col = "avg_disagreement"
    else:
        sort_col = "avg_diversity"
    
    # Sort the dataframe by the chosen metric
    analysis_df = analysis_df.sort(sort_col, descending=True)

    # Get unique prompt categories
    categories = analysis_df["prompt_category"].unique().to_list()
    
    # Initialize selected prompts list
    selected_prompts = []
    
    # Calculate how many prompts to take from each category
    prompts_per_category = max(1, num_prompts // len(categories))
    
    # Alternately select top and bottom prompts from each category
    for category in categories:
        # Filter prompts for this category
        category_prompts = analysis_df.filter(pl.col("prompt_category") == category)
        
        # Skip if category has no prompts
        if category_prompts.shape[0] == 0:
            continue
        
        # Take top prompt for this category
        top_prompt = category_prompts.head(1)["prompt_variation"][0]
        if top_prompt not in selected_prompts:
            selected_prompts.append(top_prompt)
        
        # If we need more than one prompt per category and category has enough prompts
        if prompts_per_category > 1 and category_prompts.shape[0] > 1:
            # Take bottom prompt for this category
            bottom_prompt = category_prompts.tail(1)["prompt_variation"][0]
            if bottom_prompt not in selected_prompts:
                selected_prompts.append(bottom_prompt)
    
    # If we haven't selected enough prompts, add more from overall ranking
    remaining_spots = num_prompts - len(selected_prompts)
    if remaining_spots > 0:
        # Get prompts not already selected
        remaining_prompts = analysis_df.filter(~pl.col("prompt_variation").is_in(selected_prompts))
        
        # Take half from top, half from bottom of overall list
        top_count = remaining_spots // 2 + (remaining_spots % 2)  # Add extra to top if odd
        bottom_count = remaining_spots // 2
        
        # Get top and bottom prompts
        extra_top_prompts = remaining_prompts.head(top_count)["prompt_variation"].to_list()
        extra_bottom_prompts = remaining_prompts.tail(bottom_count)["prompt_variation"].to_list()
        
        # Add to selected prompts
        selected_prompts.extend(extra_top_prompts)
        selected_prompts.extend(extra_bottom_prompts)
    
    # Trim if we somehow got too many
    if len(selected_prompts) > num_prompts:
        selected_prompts = selected_prompts[:num_prompts]
    # done
    
    return selected_prompts


def find_representative_prompts(
    df, num_prompts=15, num_iterations=100, method="random"
):
    """
    Find a set of representative prompt variations.

    Args:
        df (pl.DataFrame): DataFrame containing the evaluation results.
        num_prompts (int): Number of prompt variations to select.
        num_iterations (int): Number of iterations to run for optimization.
        method (str): Method to use for selection ("random", "pca", "kmeans", "category").

    Returns:
        tuple: (best_prompt_set, similarity_score)
    """
    # Get all unique prompt variations
    all_prompts = df.get_column("prompt_variation").unique()

    # Calculate baseline rates
    baseline_overall, baseline_per_question_model, baseline_per_model = (
        get_baseline_correct_rates(df)
    )

    best_score = 0
    best_prompt_set = []

    if method == "random":
        # Random sampling approach
        for i in range(num_iterations):
            if i % 100 == 0 and i > 0:
                print(f"Completed {i} iterations. Current best score: {best_score:.4f}")
                
            prompt_set = random.sample(all_prompts.to_list(), num_prompts)
            score = evaluate_prompt_subset(df, prompt_set)

            if score > best_score:
                best_score = score
                best_prompt_set = prompt_set
    
    elif method == "pca":
        # PCA-based selection
        print("Using PCA-based selection method...")
        best_prompt_set = pca_based_selection(df, num_prompts)
        best_score = evaluate_prompt_subset(df, best_prompt_set)
        print(f"PCA method similarity score: {best_score:.4f}")
    
    elif method == "kmeans":
        # K-means clustering based selection
        print("Using K-means clustering based selection method...")
        best_prompt_set = kmeans_based_selection(df, num_prompts)
        best_score = evaluate_prompt_subset(df, best_prompt_set)
        print(f"K-means method similarity score: {best_score:.4f}")
        
    elif method == "category":
        # Category-based selection
        print("Using category-based selection method...")
        best_prompt_set = alternating_category_selection(df, num_prompts)
        best_score = evaluate_prompt_subset(df, best_prompt_set)
        print(f"Category-based method similarity score: {best_score:.4f}")

    else:
        raise NotImplementedError(f"Method '{method}' is not implemented")

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
    # Try all methods: random sampling, PCA, K-means, and category-based
    methods = ["random", "pca", "kmeans", "category"]
    iterations = [0, 0, 0, 1]  # PCA, K-means and category methods don't use iterations

    best_score = 0
    best_method = None
    best_prompt_set = []

    for method, num_iterations in zip(methods, iterations):
        if num_iterations > 0:
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
    baseline_overall, baseline_per_question_model, baseline_per_model = (
        get_baseline_correct_rates(df)
    )
    subset_overall = calculate_correct_rate(subset_df).get_column("correct_rate")[0]
    subset_per_question_model = calculate_correct_rate(subset_df, group_keys=["question", "model_configuration"])
    subset_per_model = calculate_correct_rate(
        subset_df, group_keys=["model_configuration"]
    )

    return {
        "best_prompt_set": best_prompt_set,
        "similarity_score": best_score,
        "method": best_method,
    }


def save_comparison_to_csv(comparison_results, output_dir="comparison_results"):
    """
    Save comparison data to CSV files.
    
    Args:
        comparison_results (dict): Results dictionary from compare_subset_performance
        output_dir (str): Directory to save CSV files to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save question and model comparison data
    question_model_comparison = comparison_results["question_model_comparison"]
    question_model_csv_path = os.path.join(output_dir, "question_model_comparison.csv")
    question_model_comparison.write_csv(question_model_csv_path)
    print(f"Saved question and model comparison data to {question_model_csv_path}")
    
    # 2. Save top 5 prompts comparison data
    # Convert dictionary data to DataFrame
    top_prompts_data = []
    for model, full_rate in comparison_results["full_per_model_top_rates"].items():
        subset_rate = comparison_results["subset_per_model_top_rates"].get(model, float('nan'))
        diff = comparison_results["per_model_top_diffs"].get(model, float('nan'))
        
        top_prompts_data.append({
            "model_configuration": model,
            "full_top5_rate": full_rate,
            "subset_top5_rate": subset_rate,
            "absolute_difference": diff
        })
    
    top_prompts_df = pl.DataFrame(top_prompts_data)
    top_prompts_csv_path = os.path.join(output_dir, "top5_prompts_comparison.csv")
    top_prompts_df.write_csv(top_prompts_csv_path)
    print(f"Saved top 5 prompts comparison data to {top_prompts_csv_path}")
    
    # 3. Save bottom 5 prompts comparison data
    bottom_prompts_data = []
    for model, full_rate in comparison_results["full_per_model_bottom_rates"].items():
        subset_rate = comparison_results["subset_per_model_bottom_rates"].get(model, float('nan'))
        diff = comparison_results["per_model_bottom_diffs"].get(model, float('nan'))
        
        bottom_prompts_data.append({
            "model_configuration": model,
            "full_bottom5_rate": full_rate,
            "subset_bottom5_rate": subset_rate,
            "absolute_difference": diff
        })
    
    bottom_prompts_df = pl.DataFrame(bottom_prompts_data)
    bottom_prompts_csv_path = os.path.join(output_dir, "bottom5_prompts_comparison.csv")
    bottom_prompts_df.write_csv(bottom_prompts_csv_path)
    print(f"Saved bottom 5 prompts comparison data to {bottom_prompts_csv_path}")
    
    # 4. Save prompt set
    if "best_prompt_set" in comparison_results:
        prompt_set_df = pl.DataFrame({"prompt_variation": comparison_results["best_prompt_set"]})
    else:
        # If we're directly saving from model_comparison, we need to extract prompt set differently
        prompt_set = subset_df["prompt_variation"].unique().to_list() if 'subset_df' in locals() else []
        prompt_set_df = pl.DataFrame({"prompt_variation": prompt_set})
    
    prompt_set_csv_path = os.path.join(output_dir, "best_prompt_set.csv")
    prompt_set_df.write_csv(prompt_set_csv_path)
    print(f"Saved best prompt set to {prompt_set_csv_path}")


# +
# Usage example
# if __name__ == "__main__":
outpath = "comparison_results/sample3"
num_prompts = 35

df = load_evaluation_results()

# Calculate correct rates
print("Calculating overall correct rate...")
overall_rate = calculate_correct_rate(df)
print(f"Overall correct rate: {overall_rate.get_column('correct_rate')[0]:.4f}")

# Find representative prompts
print(f"\nFinding {num_prompts} representative prompts...")
results = find_optimal_prompt_set(df, num_prompts=num_prompts)

print("\nResults:")
print(f"Best method: {results['method']}")
print(f"Similarity score: {results['similarity_score']:.4f}")
# +
#FIXME: update doc
# I did some manual change to the result from category method, to add the top ones from PCA
df = load_evaluation_results()
manual_best_prompts = ["v_short" ,"v_reasoning" ,"v_reason_statistically" ,"v_nobody_knows" ,
                       "v_most_are_wrong_why" ,"v_help_me_guess" ,"v_exact_then_abc" ,"v_deduction_three_steps" ,
                       "v_cant_believe_it" ,"v_as_a_social_media_post" ,"v_as_a_museum_poster" ,
                       "oneshot_v_as_a_statistician" ,"occupation_prime_ministers" ,"occupation_cleaners" ,
                       "no_option_letter" ,"music_techno" ,"music_classical" ,"mental_depression" ,
                       "mental_adhd" ,"iq_low" ,"iq_high" ,"ideology_racist" ,"ideology_neoliberal" ,
                       "ideology_marxist" ,"ideology_environmentalist" ,"geo_unknown" ,"geo_soviet" ,
                       "gender_woman" ,"gender_unknown" ,"film_romantic" ,"film_comedy" ,
                       "economy_unknown" ,"economy_billionaire" ,"class_upper" ,"class_unknown" ,"source_wikipedia"]

results["best_prompt_set"] = manual_best_prompts

# Compare model performance
print("\nComparing model performance...")
subset_df = df.filter(pl.col("prompt_variation").is_in(results["best_prompt_set"]))
model_comparison = compare_subset_performance(
    df, subset_df
)

print(f"\nModel Performance Comparison:")
print(f"Average absolute difference in correct rate: {model_comparison['avg_diff']:.4f}")
print(f"Median absolute difference in correct rate: {model_comparison['median_diff']:.4f}")
print(f"25th percentile absolute difference: {model_comparison['percentile_25']:.4f}")
print(f"75th percentile absolute difference: {model_comparison['percentile_75']:.4f}")
print(f"Maximum absolute difference in correct rate: {model_comparison['max_diff']:.4f}")

# Output the optimal set of prompt variations
print(f"\nOptimal set of {num_prompts} prompt variations:")
for i, prompt_id in enumerate(results["best_prompt_set"], 1):
    print(f"{i}. {prompt_id}")

# Save comparison data to CSV files
print("\nSaving comparison data to CSV files...")
# Pass model_comparison instead of results to save_comparison_to_csv
model_comparison["best_prompt_set"] = results["best_prompt_set"]  # Add the best prompt set to model_comparison
save_comparison_to_csv(model_comparison, output_dir=outpath)
# -

calculate_correct_rate(subset_df, exclude_fail=True)

calculate_similarity_score(df, subset_df)



