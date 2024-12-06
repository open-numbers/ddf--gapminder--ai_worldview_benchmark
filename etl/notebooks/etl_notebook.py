import polars as pl
import duckdb


# first create the datapoints!
master_output = pl.read_csv('../source/Gapminder AI evaluations - Master Output.csv', schema_overrides={'Question ID': str})

# Convert column names to lowercase and connect with underscore
master_output = master_output.rename({
    col: col.lower().replace(' ', '_') for col in master_output.columns
})

# Remove '_zh' suffix from prompt_variation_id values
master_output = master_output.with_columns(
    pl.col('prompt_variation_id').str.replace('_zh$', '')
)

# Check for multiple dates per combination
date_check = (
    master_output
    .group_by(['question_id', 'prompt_variation_id', 'model_configuration_id'])
    .agg([
        pl.col('date').n_unique().alias('unique_dates'),
        pl.col('date').alias('dates')
    ])
    .filter(pl.col('unique_dates') > 1)
    .sort('unique_dates', descending=True)
)

print("Combinations with multiple dates:")
print(date_check)

# no duplicates, we can remove the language and date column
master_output = master_output.drop(['language', 'date'])

# 1. Calculate correct rates excluding 'fail' and 'n/a'
correct_rates_filtered = (
    master_output
    .filter(pl.col('result').is_in(['correct', 'wrong', 'very_wrong']))
    .group_by(['question_id', 'model_configuration_id'])
    .agg([
        pl.col('result').count().alias('total_answers'),
        (pl.col('result') == 'correct').sum().alias('correct_count')
    ])
    .with_columns([
        (pl.col('correct_count') / pl.col('total_answers') * 100).alias('correct_rates_exclude_indecisive')
    ])
    .sort(['question_id', 'model_configuration_id'])
)

print("\nCorrect rates (excluding 'fail' and 'n/a'):")
print(correct_rates_filtered)

# 2. Calculate correct rates including all results
correct_rates_all = (
    master_output
    .group_by(['question_id', 'model_configuration_id'])
    .agg([
        pl.col('result').count().alias('total_answers'),
        (pl.col('result') == 'correct').sum().alias('correct_count')
    ])
    .with_columns([
        (pl.col('correct_count') / pl.col('total_answers') * 100).alias('correct_rate_all')
    ])
    .sort(['question_id', 'model_configuration_id'])
)

print("\nCorrect rates (including all results):")
print(correct_rates_all)

# Rename columns and drop unnecessary ones before saving
correct_rates_filtered = (
    correct_rates_filtered
    .rename({
        'question_id': 'question',
        'model_configuration_id': 'model_configuration'
    })
    .drop(['total_answers', 'correct_count'])
)
correct_rates_all = (
    correct_rates_all
    .rename({
        'question_id': 'question',
        'model_configuration_id': 'model_configuration'
    })
    .drop(['total_answers', 'correct_count'])
)

# Save the results to CSV files
correct_rates_filtered.write_csv('../../ddf--datapoints--correct_rate_exclude_indecisive--by--question--model_configuration.csv')
correct_rates_all.write_csv('../../ddf--datapoints--correct_rate_all--by--question--model_configuration.csv')

correct_rates_filtered

master_output


# next create entities for model configuration
# read model list
models = pl.read_csv('../source/Gapminder AI evaluations - Models.csv')

# Convert column names to lowercase and connect with underscore
models = models.rename({
    col: col.lower().replace(' ', '_') for col in models.columns
})

# Remove rows with null model_id
models = models.filter(pl.col('model_id').is_not_null())

models

# model configurations list
model_confs = pl.read_csv('../source/Gapminder AI evaluations - Model configurations.csv')

# Convert column names to lowercase and connect with underscore
model_confs = model_confs.rename({
    col: col.lower().replace(' ', '_') for col in model_confs.columns
})

# Remove include_in_next_evaluation column
model_confs = model_confs.drop('include_in_next_evaluation')

# Join model_confs with models to include additional model information
model_confs = model_confs.join(
    models.select(['model_id', 'vendor', 'model_name', 'knowledge_cut_off_date']),
    on='model_id',
    how='left'
)

# Rename model_configuration_id to model_configuration and save as DDF entity
model_confs = model_confs.rename({'model_configuration_id': 'model_configuration'})
model_confs.write_csv('../../ddf--entities--model_configuration.csv')

model_confs

# next create the question entity
contentful_qs = pl.read_csv('../source/Contentful Questions Export - Questions.csv')

ai_eval_qs = pl.read_csv('../source/Gapminder AI evaluations - Questions.csv', schema_overrides={'Question ID': str})

# Convert column names to lowercase and connect with underscore
ai_eval_qs = ai_eval_qs.rename({
    col: col.lower().replace(' ', '_') for col in ai_eval_qs.columns
})

# Remove include_in_next_evaluation column
ai_eval_qs = ai_eval_qs.drop('include_in_next_evaluation')

# Get all unique question IDs from correct_rates_all
all_questions = correct_rates_all.select('question').unique()

# Find questions that are not in ai_eval_qs
missing_questions = all_questions.filter(
    ~pl.col('question').is_in(ai_eval_qs['question_id'].cast(pl.String))
)

# Create rows for missing questions with all columns as null except question_id
if missing_questions.height > 0:
    # Get all column names from ai_eval_qs
    columns = ai_eval_qs.columns
    
    # Create a dictionary for the new rows
    new_rows_dict = {col: [None] * missing_questions.height for col in columns}
    # Set the question_id values from missing_questions
    new_rows_dict['question_id'] = missing_questions.get_column('question').to_list()
    
    # Create the missing rows dataframe with all columns
    missing_rows = pl.DataFrame(new_rows_dict)
    
    # Append missing questions to ai_eval_qs
    ai_eval_qs = pl.concat([ai_eval_qs, missing_rows])

missing_rows
ai_eval_qs

# Create contentful_id column by removing '_t' or '_text' suffix from question_id
ai_eval_qs = ai_eval_qs.with_columns([
    pl.col('question_id').cast(pl.String)
        .str.replace('_t$|_text$', '')
        .alias('contentful_id')
])

# Group by contentful_id and fill missing information
ai_eval_qs = (
    ai_eval_qs
    .sort(['contentful_id', 'question_id'])  # Sort to ensure consistent filling
    .group_by('contentful_id')
    .map_groups(lambda df: df.fill_null(strategy='forward'))
)
ai_eval_qs.filter(pl.col('contentful_id') == '1666')

# Join with contentful_qs to get additional question information
ai_eval_qs = ai_eval_qs.join(
    contentful_qs.select([
        pl.col('globalId').cast(pl.String).alias('globalId'),
        'wrongPercentage',
        pl.col('included_in_tests_within_these_topic_ids')
    ]),
    left_on='contentful_id',
    right_on='globalId'
)

# Define the list of topics to filter
FILTERED_TOPICS = [
    'refugees',
    'population',
    'sustainable-development-misconception-study-2020',
    '2017_gapminder_test',
    'climate-misconception-study-2024',
    'sdg-world-un-goals'
]

# Rename wrongPercentage column and create sdg_world_topics and other_topics
ai_eval_qs = ai_eval_qs.rename({
    'wrongPercentage': 'human_wrong_percentage'
}).with_columns([
    pl.col('included_in_tests_within_these_topic_ids')
    .str.split(';')
    .map_elements(lambda x: next((topic for topic in x if topic.startswith('sdg-world-') and 
                                 len(topic.split('-')) == 3 and 
                                 topic.split('-')[2].isdigit() and
                                 len(topic.split('-')[2]) == 2), None),
                                 return_dtype=str)
    .alias('sdg_world_topics'),
    pl.col('included_in_tests_within_these_topic_ids')
    .str.split(';')
    .map_elements(lambda x: ';'.join([topic for topic in x if topic in FILTERED_TOPICS]))
    .alias('other_topics')
])

# Prepare and save question entity
question_entity = (
    ai_eval_qs
    .rename({
        'question_id': 'question',
        'included_in_tests_within_these_topic_ids': 'topic_list',
    })
)

# Save as DDF entity
question_entity.write_csv('../../ddf--entities--question.csv')

ai_eval_qs
contentful_qs


# create the concepts file
# Get columns from both entities
model_conf_columns = model_confs.columns
question_entity_columns = question_entity.columns

# Create a list of all unique columns
all_columns = list(set(model_conf_columns + question_entity_columns))

# Create the concepts dataframe with proper types and names
concepts_df = pl.DataFrame({
    'concept': all_columns + ['correct_percentage_all', 'correct_percentage_exclude_indecisive'],
    'concept_type': ['string'] * len(all_columns) + ['measure', 'measure'],
    'name': [concept.replace('_', ' ').title() for concept in all_columns] + 
           ['Correct Percentage (all answers)', 'Correct Percentage (excluding indecisive answers)']
})

# Update entity domains
concepts_df = concepts_df.with_columns([
    pl.when(pl.col('concept') == 'question')
    .then(pl.lit('entity_domain'))
    .when(pl.col('concept') == 'model_configuration')
    .then(pl.lit('entity_domain'))
    .otherwise(pl.col('concept_type'))
    .alias('concept_type')
])

# Save as DDF concepts
concepts_df.write_csv('../../ddf--concepts.csv')

# Check specific combinations after all processing
questions_to_check = ['1757', '11', '1764', '59', '1632', '1546', '1508', '1', '32', '1601', '1593', '814']
models_to_check = ['mc036', 'mc037', 'mc038', 'mc039', 'mc040']

filtered_rates = (
    correct_rates_all
    .filter(
        pl.col('question').cast(pl.Utf8).is_in(questions_to_check) & 
        pl.col('model_configuration').is_in(models_to_check)
    )
    .sort(['question', 'model_configuration'])
)

print("\nFiltered correct rates for specific combinations:")
print(filtered_rates)

# Calculate missing combinations
all_combinations = pl.DataFrame({
    'question': questions_to_check * len(models_to_check),
    'model_configuration': [model for model in models_to_check for _ in questions_to_check]
})

missing_combinations = (
    all_combinations.join(
        filtered_rates.select(['question', 'model_configuration']),
        on=['question', 'model_configuration'],
        how='anti'
    )
    .sort(['question', 'model_configuration'])
)

print("\nMissing combinations:")
print(missing_combinations)

# Check if any questions from our checking list are missing from question entities
missing_from_entities = (
    pl.DataFrame({'question': questions_to_check})
    .join(
        question_entity.select('question'),
        on='question',
        how='anti'
    )
    .sort('question')
)

print("\nQuestions from checking list that are missing from question entities:")
print(missing_from_entities)

# filter results
master_output
master_output.filter(
    pl.col('question_id').is_in(questions_to_check)
)


# check average correct rate for xai
correct_rates_filtered.filter(
    pl.col('model_configuration') == 'mc036'
).select(
    pl.col('correct_rates_exclude_indecisive').mean()
)
