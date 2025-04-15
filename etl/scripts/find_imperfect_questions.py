import pandas as pd
import argparse

def find_imperfect_questions(model_config_id):
    # Read the input CSV file
    df = pd.read_csv(
      'ddf--datapoints--evaluation_result--by--question--model_configuration--prompt_variation.csv',
      dtype={"question": str}
      )

    # Filter for the specified model configuration
    model_df = df[df['model_configuration'] == model_config_id]

    # Get unique question IDs for this model configuration
    question_ids = model_df['question'].unique()

    imperfect_questions = []

    for qid in question_ids:
        # Get all evaluations for this question and model config
        q_df = model_df[model_df['question'] == qid]

        # Filter out 'fail' and 'n/a' evaluations
        valid_df = q_df[~q_df['evaluation_result'].isin(['fail', 'n/a'])]

        # Skip if no valid evaluations remain
        if len(valid_df) == 0:
            continue

        # Check if all valid evaluations are 'correct'
        if not all(valid_df['evaluation_result'] == 'correct'):
            imperfect_questions.append(qid)

    return imperfect_questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find questions where a model configuration did not achieve 100% correctness')
    parser.add_argument('--model-config-id', required=True, help='Model configuration ID to analyze')
    args = parser.parse_args()

    questions = find_imperfect_questions(args.model_config_id)

    # Print results, one per line
    for q in questions:
        print(q)
