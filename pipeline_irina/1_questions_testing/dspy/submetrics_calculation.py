import re
import numpy as np
import spacy
from mind.prompter.prompter import Prompter
from transformers import pipeline
from bert_score import score
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import time


print("Loading SBERT model...")

sbert_model = SentenceTransformer(
     "intfloat/multilingual-e5-large"
)


print("Loading NLI model...")

nli = pipeline(
    "text-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    top_k=None
)

def noun_missing_ratio(question, passage):

    q_doc = nlp(question)
    p_doc = nlp(passage)

    q_nouns = {
        token.lemma_.lower()
        for token in q_doc
        if token.pos_ in ["NOUN", "PROPN"]
    }

    p_nouns = {
        token.lemma_.lower()
        for token in p_doc
        if token.pos_ in ["NOUN", "PROPN"]
    }

    if len(q_nouns) == 0:
        return 0.0

    missing = q_nouns - p_nouns

    return len(missing) / len(q_nouns)

def sbert_cosine_similarity(
    question,
    passage
):

    embeddings = sbert_model.encode(
        [question, passage],
        convert_to_tensor=True
    )

    similarity = cos_sim(
        embeddings[0],
        embeddings[1]
    )

    return float(similarity.item())

def entailment_score(question, passage):

    result = nli(
        {
            "text": passage,
            "text_pair": question
        }
    )

    scores = {
        item["label"].lower(): item["score"]
        for item in result
    }
    return scores


bertscorer = BERTScorer(
    lang="en",
    rescale_with_baseline=True
)

def bertscore_similarity(question, passage):

    P, R, F1 = bertscorer.score(
        [question],
        [passage]
    )

    return float(F1[0])


nlp = spacy.load("en_core_web_sm")

judge_prompter = Prompter(
    model_type="qwen3.6:27b",
    config_path="config/config_i.yaml",
    temperature=0,
)

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())



def lexical_overlap(question, context):
    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(context))

    if len(q_tokens) == 0:
        return 0

    return len(q_tokens & c_tokens) / len(q_tokens)

def jaccard_overlap(question, context):
    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(context))

    if len(q_tokens) == 0:
        return 0

    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)


def ngram_overlap(question, context, n=2):
    q_tokens = tokenize(question)
    c_tokens = tokenize(context)

    if len(q_tokens) < n:
        return 0

    q_ngrams = set(zip(*[q_tokens[i:] for i in range(n)]))
    c_ngrams = set(zip(*[c_tokens[i:] for i in range(n)]))

    if len(q_ngrams) == 0:
        return 0

    return len(q_ngrams & c_ngrams) / len(q_ngrams)






def ask_llm_judge(criterion, anchor_passage,question):
    prompt = f"""
        You are an evaluator for question generation.

        Criterion:
        {criterion}

        Passage:
        {anchor_passage}

        Question:
        {question}

        """

    response, _ = judge_prompter.prompt(
        question=prompt,
        dry_run=False,
    )

    response = response.strip()

    match = re.search(
        r"(?:0(?:\.25|\.5|\.75)?|1(?:\.0)?)",
        response
    )

    if match:
        score = float(match.group())
    else:
        score = 0.5

    score = max(0.0, min(score, 1.0))

    return score

def compute_llm_judges(anchor_passage, question):

    atomized = ask_llm_judge(
        criterion = """
           Task: Determine whether the following question is ATOMIC, meaning it asks about a single, specific fact that can be extracted from the passage.

            Definition:
            - 1: the question can be answered by checking a single statement in the passage.
            - 0: the question combines two or more distinct statements, conditions, or facts.

            Respond with ONLY one value:

            1
            or
            0

            Do not add any explanation.
            """
        ,

        anchor_passage=anchor_passage,
        question=question
    )

    answerable = ask_llm_judge(
        criterion="""
            Task: Determine whether the following question is ANSWERABLE across documents discussing the same topic.

            Definition:
            - 1: the question asks about information that is general enough to appear in multiple documents on the same topic.
            - 0: the question depends on highly specific details such as exact names, numbers, percentages, dates, or unique recommendations.

            Respond with ONLY one value:

            1
            or
            0

            Do not add any explanation.
            """   
        ,

        anchor_passage=anchor_passage,
        question=question
    )

    contextualized = ask_llm_judge(
        criterion="""
           Task: Determine whether the following question is CONTEXTUALIZED.

            Definition:
            - 1: the question can be understood without reading the original passage.
            - 0: the question contains references, omissions, or vague expressions that require the passage.

            Respond with ONLY one value:

            1
            or
            0

            Do not add any explanation.
            """
        ,

        anchor_passage=anchor_passage,
        question=question
    )

    subordinate = ask_llm_judge(
        criterion="""
            Task: Determine whether the following question has a SIMPLE structure.

            Definition:
            - 1: the question contains a single request and does not rely on multiple alternatives, conjunctions, subordinate clauses, or lists of options.
            - 0: the question contains multiple alternatives, multiple options, conjunctions, or subordinate clauses that introduce several possible conditions.

            Respond with ONLY one value:

            1
            or
            0

            Do not add any explanation.
            """
        ,
        anchor_passage=anchor_passage,
        question=question
    )

    return {
        "atomized": atomized,
        "answerable": answerable,
        "contextualized": contextualized,
        "subordinate": subordinate
    }



def extract_features(row):

    question = row["question"]
    context = row["anchor_passage"]
    judges = compute_llm_judges(anchor_passage=context, question=question)

    nli_scores = entailment_score(question,context)

    features = {
        #"atomized": row["atomized"], #las mias 0 si no lo cumple, 1 si si lo cumple
        #"answerable": row["answerable"], #las mias 0 si no lo cumple, 1 si si lo cumple
        #"contextualized": row["contextualized"], #las mias 0 si no lo cumple, 1 si si lo cumple

        "atomized": judges["atomized"], #llm as a judge

        "answerable": judges["answerable"], #llm as a judge

        "contextualized": judges["contextualized"], #llm as a judge

        "subordinate": judges["subordinate"], #llm as a judge

        "noun_missing_ratio": noun_missing_ratio(question, context),

        "lexical_overlap": lexical_overlap(question, context),

        "jaccard_overlap": jaccard_overlap(question, context),

        "2gram_overlap": ngram_overlap(question, context, n=2),

        "3gram_overlap": ngram_overlap(question, context, n=3),

        "BertSCORE": bertscore_similarity(question, context), #de momento asi está bien, pero recibe de parámetro el idioma.

        "Cosine_SBERT": sbert_cosine_similarity(question, context),

        "entailment": nli_scores["entailment"],

        "neutral": nli_scores["neutral"],

        "contradiction": nli_scores["contradiction"]

        #"nli_margin":
        #    nli_scores["entailment"]- nli_scores["contradiction"]

    }

    return features
