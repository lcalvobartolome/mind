# ============================================================
# IMPORTS
# ============================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

import re
import joblib
import pandas as pd
import numpy as np
import dspy

from pathlib import Path

from mind.prompter.prompter import Prompter



DATASET_PATH = Path(
    "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated.xlsx"
)

MODEL_PATH = Path(
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/definitivo/question_quality.pkl"
)

CONFIG_PATH = Path(
    "config/config_i.yaml"
)

OUTPUT_DIR = Path(
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/dspy"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_excel(
    DATASET_PATH,
    usecols=[
        "anchor_passage"
    ]
)

print(f"Loaded {len(df)} examples")

saved = joblib.load(MODEL_PATH)


if isinstance(saved, dict):

    regression_model = saved["model"]
    regression_features = saved["features"]

else:

    regression_model = saved

    regression_features = [
        "subordinate",
        "answerable",
        "jaccard"
    ]

print("Regression model loaded")


generator_lm = dspy.LM(
    "ollama_chat/qwen3.6:27b",
    api_base="http://kumo01.tsc.uc3m.es:11434",
    temperature=0,
)

dspy.configure(lm=generator_lm)

judge_prompter = Prompter(
    model_type="qwen3.6:27b",
    config_path=CONFIG_PATH,
    temperature=0,
)


print("Prompters ready")

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        str(text).lower()
    )


def jaccard_overlap(question, context):

    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(context))

    if len(q_tokens) == 0:
        return 0.0

    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)



def regression_score(
    subordinate,
    answerable,
    jaccard,
):
    """
    Returns the probability that the generated question
    is a GOOD question.
    """

    X = pd.DataFrame(
        [[
            subordinate,
            answerable,
            jaccard
        ]],
        columns=regression_features
    )

    probability = regression_model.predict_proba(X)[0, 1]

    return float(probability)

class GenerateQuestion(dspy.Signature):
    """
    Generate ONE question.

    The question must:

    - ask only one thing
    - be answerable from the passage
    - avoid specific dates, names, numbers or particular cases whenever possible
    - be useful for discrepancy detection
    - ask about information likely to appear across multiple documents discussing the same topic
    - return only the question
    """

    passage = dspy.InputField()
    question = dspy.OutputField()



class QuestionGenerator(dspy.Module):

    def __init__(self):

        super().__init__()

        self.generate = dspy.ChainOfThought(
            GenerateQuestion
        )

    def forward(self, passage):

        prediction = self.generate(
            passage=passage
        )

        return prediction


def ask_llm_judge(
    criterion,
    anchor_passage,
    question,
):
    """
    Executes one LLM-as-a-Judge prompt.
    Returns only 0 or 1.
    """

    prompt = f"""
You are an evaluator for question generation.

Criterion

{criterion}

Passage

{anchor_passage}

Question

{question}

Respond ONLY with

1

or

0
"""

    response, _ = judge_prompter.prompt(
        question=prompt,
        dry_run=False,
    )

    response = response.strip()

    matches = re.findall(r"\b[01]\b", response)

    if matches:
        return int(matches[-1])


    return 0


SUBORDINATE_PROMPT = """
Task: Determine whether the following question has a SIMPLE structure.

Definition:

- 1: the question contains a single request and does not rely on multiple alternatives,
conjunctions,
subordinate clauses,
or lists.

- 0: the question contains several alternatives,
multiple conditions,
or subordinate clauses.

Respond ONLY

1

or

0
"""


ANSWERABLE_PROMPT = """
Task: Determine whether the following question is ANSWERABLE across documents discussing
the same topic.

Definition:

- 1: the question asks about information likely to appear in several documents.

- 0: the question depends on unique details such as dates,
numbers,
names,
percentages,
or recommendations.

Respond ONLY

1

or

0
"""


def compute_judges(
    passage,
    question,
):

    subordinate = ask_llm_judge(
        criterion=SUBORDINATE_PROMPT,
        anchor_passage=passage,
        question=question,
    )

    answerable = ask_llm_judge(
        criterion=ANSWERABLE_PROMPT,
        anchor_passage=passage,
        question=question,
    )

    return {
        "subordinate": subordinate,
        "answerable": answerable,
    }

def question_quality_metric(
    example,
    prediction,
    trace=None,
):
    """
    Reward used by DSPy.

    The higher the probability returned by the logistic
    regression model, the better.
    """

    generated_question = prediction.question

    judges = compute_judges(
        passage=example.passage,
        question=generated_question,
    )

    jaccard = jaccard_overlap(
        generated_question,
        example.passage,
    )

    score = regression_score(
        subordinate=judges["subordinate"],
        answerable=judges["answerable"],
        jaccard=jaccard,
    )

    return float(score)



examples = []

for _, row in df.iterrows():

    ex = dspy.Example(
        passage=row["anchor_passage"]
    ).with_inputs("passage")

    examples.append(ex)

print(f"Created {len(examples)} DSPy examples")



np.random.seed(42)

np.random.shuffle(examples)

n = len(examples)

train_size = int(0.70 * n)

dev_size = int(0.15 * n)

trainset = examples[:train_size]

devset = examples[
    train_size:train_size + dev_size
]

testset = examples[
    train_size + dev_size:
]

print()

print(f"Train : {len(trainset)}")

print(f"Dev   : {len(devset)}")

print(f"Test  : {len(testset)}")



generator = QuestionGenerator()



print("\nTesting metric...\n")

prediction = generator(
    passage=trainset[0].passage
)

metric_value = question_quality_metric(
    trainset[0],
    prediction,
)

print("Generated question:\n")

print(prediction.question)

print()

print(f"Regression score = {metric_value:.4f}")


from dspy.teleprompt import MIPROv2

print("\nStarting DSPy optimization...\n")

optimizer = MIPROv2(
    metric=question_quality_metric,
    auto="medium",          # light medium heavy
    num_threads=4,         
    seed=42,
    verbose=True,
)

optimized_generator = optimizer.compile(
    student=generator,
    trainset=trainset,
    valset=devset,
)

print("\nOptimization finished.")

OUTPUT_PROGRAM = OUTPUT_DIR / "optimized_generator_qwen_medium.json"

optimized_generator.save(str(OUTPUT_PROGRAM))

print(f"\nOptimized program saved to:\n{OUTPUT_PROGRAM}")


print("\nEvaluating on test set...\n")

scores = []

for example in testset:

    prediction = optimized_generator(
        passage=example.passage
    )

    score = question_quality_metric(
        example,
        prediction,
    )

    scores.append(score)

mean_score = float(np.mean(scores))

print(f"\nMean reward on test set: {mean_score:.4f}")


print("\nExample generations\n")

for i, example in enumerate(testset[:5]):

    prediction = optimized_generator(
        passage=example.passage
    )

    print("=" * 80)
    print(f"Example {i+1}\n")
    print("PASSAGE\n")
    print(example.passage)
    print("\nQUESTION\n")
    print(prediction.question)

    score = question_quality_metric(
        example,
        prediction,
    )

    print(f"\nReward = {score:.4f}")

print("\nDone.")