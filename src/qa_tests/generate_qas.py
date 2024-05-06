from qa_metrics.prompt_llm import CloseLLM
import pandas as pd
import pathlib
from dotenv import load_dotenv
import os
import json
import requests
import logging
import time

logging.basicConfig(level=logging.DEBUG)


def make_question_rosie(question, lang):
    
    question_data = {
        "text": question,
        "userUID": "lorena-trial",
        "lang": lang
    }
    
    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    url_post = "https://rosie.umiacs.umd.edu/ask"
    post_response = requests.post(url_post, data=question_data, headers=header)
    post_response_json = post_response.content
    
    return eval(post_response_json)["text"]#, eval(post_response_json)["source_document_url"]

def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()

    return file_contents


path_env = pathlib.Path(os.getcwd()).parent.parent / '.env'

load_dotenv(path_env)
api_key = os.getenv("OPENAI_API_KEY")
gpt_model = CloseLLM()
gpt_model.set_openai_api_key(api_key)

prompt_template = load_prompt_template("./promt.txt")

all_questions = []
iters = 10
for iter_ in range(iters):
    time.sleep(1)
    llm_response = gpt_model.prompt_gpt(
        prompt=prompt_template, model_engine='gpt-3.5-turbo-instruct', temperature=0.1, max_tokens=3000
    )
    questions = eval(llm_response)
    all_questions.append(questions)

def flatten(xss): return [x for xs in xss for x in xs]
all_questions = flatten(all_questions)

df = pd.DataFrame(
    {
        "qa_id": range(len(all_questions)),
        "en_question": [el[0] for el in all_questions],
        "es_question": [el[1] for el in all_questions]
    }
)

df["en_answer"] = df["en_question"].apply(lambda x: make_question_rosie(x, "en"))
df["es_answer"] = df["es_question"].apply(lambda x: make_question_rosie(x, "es"))

df.to_csv(f"sample_qas_{iters}.csv", index=False)

import pdb; pdb.set_trace()