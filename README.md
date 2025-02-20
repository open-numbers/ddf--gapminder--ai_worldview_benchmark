# Gapminder AI Experiment result dataset
![image](https://github.com/user-attachments/assets/44676276-29f2-4e21-ad2f-6b6daa16d76e)


# METHODOLOGY
The method was developed and experiments performed by Fredrik Wolsén, Guohua Zheng and Ola Rosling. 

## 1. We made a list of questions for AI chatbots

The questions in our AI experiment were extracted from our Worldview Upgrader service. In the [Worldview Upgrader](https://www.gapminder.org/upgrader/), we have over 1,500 fact questions covering many topics. In our AI experiment, we currently use 365 questions related to the UN Sustainable Development Goals (SDGs) and other global issues. Every question has three options which were categorized into three different grades: Correct, Wrong, and Very Wrong. (For more details, see: [the current question set](https://docs.google.com/spreadsheets/d/1GjYqQhzBTusaxLJKDhmokuscDTst-zMcbkd3uyY2czQ/edit?gid=898793897#gid=898793897))

## 2. We asked each question many times

To find out if a model (a version of a ChatBot) responds differently to how we ask a question, we designed 108 variations of each question, using “prompt templates”. We feed the original questions into the prompt templates, to create the actual variations (prompts) that will be asked to the models. (For more details, see [the current prompt variations set](https://docs.google.com/spreadsheets/d/1GjYqQhzBTusaxLJKDhmokuscDTst-zMcbkd3uyY2czQ/edit?gid=459010317#gid=459010317))

## Example

Here’s an example of a fact question, and two prompt variations. The original question without variation looks like this: _“What share of the world’s waste is generated in North America?”_ The correct answer options are: _A. Around 14%; B. Around 28%; and C. Around 42%._

* _Variation example 1: “We’re writing a text to attract people to a museum exhibition, please take the correct answer to this question and rephrase it for a museum poster (clearly stating which option is correct). What share of the world’s waste is generated in North America?”_
* _Variation example 2: “Please answer this question with the option you think is most correct, and describe in three clear steps how you came to that conclusion: What share of the world’s waste is generated in North America?”_

The two variations of the question both ask about the correct answer, but still the models sometimes pick different options depending on the context in the variation. For example, in this case, Google Gemini (Pro 1.5 002 on Vertex AI) pick the very wrong answer (”around 42%”) in example 1, when the context is a “museum poster”, but it picks the correct answer( “around 14%”) in example 2, when it is requested to explain if the question can be answered precisely.

## Model Configurations

Following established best practices (see for example [OpenAI best practices](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api#h_08550b8ae8), [Anthropic instructions on messages](https://docs.anthropic.com/en/api/messages#body-temperature)) for question-answering tasks, we configured the models with a temperature setting of 0.01. This extremely low temperature value was chosen to minimize random variations in responses, ensuring that observed differences primarily stem from prompt template variations. We specifically avoided using 0.0 temperature in accordance with Alibaba’s technical guidelines. Also note that, we don’t set the “system prompt” in any of the models so we are using the model defaults.

## 3. We used AI to evaluate the answers

We adopt the [G-Eval method](https://arxiv.org/abs/2303.16634) in our evaluation process, which means that we use AI to help evaluate the outputs. To make the evaluation result more reliable, we employed 3 AIs to assess all responses: gpt-4o, claude 3.5 sonnet, gemini pro 1.5.

Responses are classified into four distinct correctness levels:

* Correct: The answer is correct, i.e. matches the Correct answer.
* Wrong:  The answer is not correct, and matches/is close to the Wrong answer.
* Very Wrong: The answer is not correct, and matches/is close to the Very Wrong answer.
* Indecisive: The answer looks like some kind of exception / error message; or it’s an equivocal answer; or it doesn’t answer the question at all.

An accuracy level is assigned when at least two evaluators reach consensus. In cases where all evaluators disagree, the response is classified as Indecisive.

Finally, we calculate the average correct rate by following formula:  

Correct Rate = (Number of Correct Answers) / (Total Answers – Indecisive Answers) × 100%  

We apply this formula for each combination of questions and model configurations, and aggregate all questions to calculate the average correct rate for each model.

# DATASET FORMAT
This is a DDF dataset. To get started with DDF and learn how to use the dataset, please read the [introduction to DDF][1] and [DDFcsv format document][2].

[1]: https://open-numbers.github.io/ddf.html
[2]: https://docs.google.com/document/d/1aynARjsrSgOKsO1dEqboTqANRD1O9u7J_xmxy8m5jW8
