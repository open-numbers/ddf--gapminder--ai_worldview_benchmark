import polars as pl
import duckdb


# first create the datapoints!
master_output = pl.read_csv(
    "../source/Gapminder AI evaluations - Master Output.csv",
    schema_overrides={"Question ID": str},
)

# Convert column names to lowercase and connect with underscore
master_output = master_output.rename(
    {col: col.lower().replace(" ", "_") for col in master_output.columns}
)

# Remove '_zh' suffix from prompt_variation_id values
master_output = master_output.with_columns(
    pl.col("prompt_variation_id").str.replace("_zh$", "")
)

# Check for multiple dates per combination
date_check = (
    master_output.group_by(
        ["question_id", "prompt_variation_id", "model_configuration_id"]
    )
    .agg(
        [pl.col("date").n_unique().alias("unique_dates"), pl.col("date").alias("dates")]
    )
    .filter(pl.col("unique_dates") > 1)
    .sort("unique_dates", descending=True)
)

print("Combinations with multiple dates:")
print(date_check)

# Keep only the latest datapoint for each combination
master_output = (
    master_output.sort(
        ["question_id", "prompt_variation_id", "model_configuration_id", "date"]
    )
    .group_by(["question_id", "prompt_variation_id", "model_configuration_id"])
    .agg(pl.all().last())  # Keep the last (most recent) record for each group
)

# Now we can remove the language and date columns as they're no longer needed
master_output = master_output.drop(["language", "date"])

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

print("\nAll rates calculated:")
print(rates)

# Check for NaN values in rates
nan_rows = rates.filter(
    pl.col("correct_rate").is_nan()
    | pl.col("wrong_rate").is_nan()
    | pl.col("very_wrong_rate").is_nan()
    | pl.col("indecisive_rate").is_nan()
)

if nan_rows.height > 0:
    print("\nWARNING: Found rows with NaN values:")
    print(nan_rows)
    print("\nOriginal data for these questions:")
    print(master_output.filter(pl.col("question_id").is_in(nan_rows["question_id"])))

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
    (
        rates.select(["question", "model_configuration", rate_type]).write_csv(
            f"../../ddf--datapoints--{rate_type}--by--question--model_configuration.csv"
        )
    )

rates

master_output


# next create entities for model configuration
# read model list
models = pl.read_csv("../source/Gapminder AI evaluations - Models.csv")

# Convert column names to lowercase and connect with underscore
models = models.rename({col: col.lower().replace(" ", "_") for col in models.columns})

# Remove rows with null model_id
models = models.filter(pl.col("model_id").is_not_null())

models

# model configurations list
model_confs = pl.read_csv(
    "../source/Gapminder AI evaluations - Model configurations.csv"
)

# Convert column names to lowercase and connect with underscore
model_confs = model_confs.rename(
    {col: col.lower().replace(" ", "_") for col in model_confs.columns}
)

# Remove include_in_next_evaluation column
model_confs = model_confs.drop("include_in_next_evaluation")

# Join model_confs with models to include additional model information
model_confs = model_confs.join(
    models.select(["model_id", "vendor", "model_name", "knowledge_cut_off_date"]),
    on="model_id",
    how="left",
)

# Add is_latest_model column
latest_models = ["mc030", "mc036", "mc037", "mc038", "mc039", "mc040"]
model_confs = model_confs.with_columns(
    [
        pl.col("model_configuration_id")
        .is_in(latest_models)
        .map_elements(lambda x: "TRUE" if x else "FALSE", return_dtype=str)
        .alias("is--latest_model")
    ]
)


# Rename model_configuration_id to model_configuration and save as DDF entity
model_confs = model_confs.rename({"model_configuration_id": "model_configuration"})
model_confs.write_csv("../../ddf--entities--model_configuration.csv")

model_confs

# next create the question entity
contentful_qs = pl.read_csv("../source/Contentful Questions Export - Questions.csv")

ai_eval_qs = pl.read_csv(
    "../source/Gapminder AI evaluations - Questions.csv",
    schema_overrides={"Question ID": str},
)

# Convert column names to lowercase and connect with underscore
ai_eval_qs = ai_eval_qs.rename(
    {col: col.lower().replace(" ", "_") for col in ai_eval_qs.columns}
)

# Remove include_in_next_evaluation column
ai_eval_qs = ai_eval_qs.drop("include_in_next_evaluation")

# Get all unique question IDs from rates
all_questions = rates.select("question").unique()

# Find questions that are not in ai_eval_qs
missing_questions = all_questions.filter(
    ~pl.col("question").is_in(ai_eval_qs["question_id"].cast(pl.String))
)

# Create rows for missing questions with all columns as null except question_id
if missing_questions.height > 0:
    # Get all column names from ai_eval_qs
    columns = ai_eval_qs.columns

    # Create a dictionary for the new rows
    new_rows_dict = {col: [None] * missing_questions.height for col in columns}
    # Set the question_id values from missing_questions
    new_rows_dict["question_id"] = missing_questions.get_column("question").to_list()

    # Create the missing rows dataframe with all columns
    missing_rows = pl.DataFrame(new_rows_dict)

    # Append missing questions to ai_eval_qs
    ai_eval_qs = pl.concat([ai_eval_qs, missing_rows])

missing_rows
ai_eval_qs

# Create contentful_id column by removing '_t' or '_text' suffix from question_id
ai_eval_qs = ai_eval_qs.with_columns(
    [
        pl.col("question_id")
        .cast(pl.String)
        .str.replace("_t$|_text$", "")
        .alias("contentful_id")
    ]
)

# Group by contentful_id and fill missing information
ai_eval_qs = (
    ai_eval_qs.sort(
        ["contentful_id", "question_id"]
    )  # Sort to ensure consistent filling
    .group_by("contentful_id")
    .map_groups(lambda df: df.fill_null(strategy="forward"))
)
ai_eval_qs.filter(pl.col("contentful_id") == "1666")

# Join with contentful_qs to get additional question information
ai_eval_qs = ai_eval_qs.join(
    contentful_qs.select(
        [
            pl.col("globalId").cast(pl.String).alias("globalId"),
            "wrongPercentage",
            pl.col("included_in_tests_within_these_topic_ids"),
        ]
    ),
    left_on="contentful_id",
    right_on="globalId",
)

# Define the list of topics to filter
FILTERED_TOPICS = [
    "refugees",
    "population",
    "sustainable-development-misconception-study-2020",
    "2017_gapminder_test",
    "climate-misconception-study-2024",
    "sdg-world-un-goals",
]

# Rename wrongPercentage column and create sdg_world_topics and other_topics
ai_eval_qs = ai_eval_qs.rename(
    {"wrongPercentage": "human_wrong_percentage"}
).with_columns(
    [
        pl.col("included_in_tests_within_these_topic_ids")
        .str.split(";")
        .map_elements(
            lambda x: next(
                (
                    topic
                    for topic in x
                    if topic.startswith("sdg-world-")
                    and len(topic.split("-")) == 3
                    and topic.split("-")[2].isdigit()
                    and len(topic.split("-")[2]) == 2
                ),
                None,
            ),
            return_dtype=str,
        )
        .alias("sdg_world_topics"),
        pl.col("included_in_tests_within_these_topic_ids")
        .str.split(";")
        .map_elements(
            lambda x: ";".join(
                [topic for topic in x if topic in FILTERED_TOPICS], return_dtype=pl.Utf8
            )
        )
        .alias("other_topics"),
    ]
)

# TODO: now read the Question options
# Read question options
question_options = pl.read_csv(
    "../source/Gapminder AI evaluations - Question options.csv"
)

# Convert column names to lowercase and connect with underscore
question_options = question_options.rename(
    {col: col.lower().replace(" ", "_") for col in question_options.columns}
)

# Create pivot table for answers based on correctness
answers_pivot = (
    question_options.filter(
        pl.col("language") == "en-US"
    )  # Filter for English answers only
    .with_columns(
        pl.col("question_id").cast(pl.Utf8),
        pl.col("question_option").alias("answer"),
        pl.col("correctness_of_answer_option").alias("correctness"),
    )
    .group_by("question_id")
    .agg(
        [
            pl.col("answer")
            .filter(pl.col("correctness") == 1)
            .first()
            .alias("correct_answer"),
            pl.col("answer")
            .filter(pl.col("correctness") == 2)
            .first()
            .alias("wrong_answer"),
            pl.col("answer")
            .filter(pl.col("correctness") == 3)
            .first()
            .alias("very_wrong_answer"),
        ]
    )
)

answers_pivot.filter(pl.col("question_id") == "33")
question_options.filter(pl.col("question_id") == 33)

# Prepare question entity
question_entity = ai_eval_qs.rename(
    {
        "question_id": "question",
        "included_in_tests_within_these_topic_ids": "topic_list",
    }
)

# Join question options with question entity
question_entity = question_entity.join(
    answers_pivot, left_on="contentful_id", right_on="question_id", how="left"
)

question_entity

# Save as DDF entity
question_entity.write_csv("../../ddf--entities--question.csv")

ai_eval_qs
contentful_qs


# create the concepts file
# Get columns from both entities
model_conf_columns = model_confs.columns
question_entity_columns = question_entity.columns

# Create a list of all unique columns and transform them
all_columns = list(set(model_conf_columns + question_entity_columns))
all_columns = [col[4:] if col.startswith("is--") else col for col in all_columns]
all_columns.extend(["name", "domain"])

# Create concepts dataframe for string columns
string_concepts = pl.DataFrame(
    {
        "concept": all_columns,
        "concept_type": ["string"] * len(all_columns),
        "name": [concept.replace("_", " ").title() for concept in all_columns],
    }
)

# Create concepts dataframe for measure columns
measure_concepts = pl.DataFrame(
    {
        "concept": ["correct_rate", "wrong_rate", "very_wrong_rate", "indecisive_rate"],
        "concept_type": ["measure"] * 4,
        "name": [
            "Correct Rate (excluding indecisive answers)",
            "Wrong Rate (excluding indecisive answers)",
            "Very Wrong Rate (excluding indecisive answers)",
            "Indecisive Rate",
        ],
    }
)

# Combine the two concept dataframes
concepts_df = pl.concat([string_concepts, measure_concepts])

# Update entity domains
concepts_df = concepts_df.with_columns(
    [
        pl.when(pl.col("concept") == "question")
        .then(pl.lit("entity_domain"))
        .when(pl.col("concept") == "model_configuration")
        .then(pl.lit("entity_domain"))
        .when(pl.col("concept") == "latest_model")
        .then(pl.lit("entity_set"))
        .otherwise(pl.col("concept_type"))
        .alias("concept_type")
    ]
)

# Set domain for latest_model
concepts_df = concepts_df.with_columns(
    [
        pl.when(pl.col("concept") == "latest_model")
        .then(pl.lit("model_configuration"))
        .otherwise(None)
        .alias("domain")
    ]
)

# Save as DDF concepts
concepts_df.write_csv("../../ddf--concepts.csv")

# Check specific combinations after all processing
questions_to_check = [
    "1757",
    "11",
    "1764",
    "59",
    "1632",
    "1546",
    "1508",
    "1",
    "32",
    "1601",
    "1593",
    "814",
]
models_to_check = ["mc036", "mc037", "mc038", "mc039", "mc040"]

filtered_rates = rates.filter(
    pl.col("question").cast(pl.Utf8).is_in(questions_to_check)
    & pl.col("model_configuration").is_in(models_to_check)
).sort(["question", "model_configuration"])

print("\nFiltered correct rates for specific combinations:")
print(filtered_rates)

# Calculate missing combinations
all_combinations = pl.DataFrame(
    {
        "question": questions_to_check * len(models_to_check),
        "model_configuration": [
            model for model in models_to_check for _ in questions_to_check
        ],
    }
)

missing_combinations = all_combinations.join(
    filtered_rates.select(["question", "model_configuration"]),
    on=["question", "model_configuration"],
    how="anti",
).sort(["question", "model_configuration"])

print("\nMissing combinations:")
print(missing_combinations)

# Check if any questions from our checking list are missing from question entities
missing_from_entities = (
    pl.DataFrame({"question": questions_to_check})
    .join(question_entity.select("question"), on="question", how="anti")
    .sort("question")
)

print("\nQuestions from checking list that are missing from question entities:")
print(missing_from_entities)

# filter results
master_output
master_output.filter(pl.col("question_id").is_in(questions_to_check))


# check average rates for xai
rates.filter(pl.col("model_configuration") == "mc036").select(
    [
        pl.col("correct_rate").mean().alias("avg_correct_rate"),
        pl.col("wrong_rate").mean().alias("avg_wrong_rate"),
        pl.col("very_wrong_rate").mean().alias("avg_very_wrong_rate"),
        pl.col("indecisive_rate").mean().alias("avg_indecisive_rate"),
    ]
)
