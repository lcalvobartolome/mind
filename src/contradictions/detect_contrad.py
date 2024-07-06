import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune
import os
from dotenv import load_dotenv
import pathlib
import re
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import dsp
import numpy as np
from scipy import sparse
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import faiss
import json
#from src.qa_system.checker.lin_qa_checker import LinQAChecker

class QAChecker(dspy.Signature):
    """Evaluate the relationship between two ANSWER1 and ANSWER2 to the given QUESTION.
    """
    
    QUESTION = dspy.InputField(desc="The question for which the answers are being evaluated.")
    ANSWER1 = dspy.InputField(desc="The first answer to the question, used as the reference point.")
    ANSWER2 = dspy.InputField(desc="The second answer to the question, evaluated against the first answer.")
    LABEL = dspy.OutputField(desc="The relationship of ANSWER2 to ANSWER1: CONSISTENT or CONTRADICTORY")
    RATIONALE = dspy.OutputField(desc="The explanation or justification for the assigned LABEL.")
    
    
class LinQAChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.checker = dspy.ChainOfThought(QAChecker)
    
    def forward(self, question, gold_answer, gen_answer):        
        contrad = None
        if gen_answer:
            try:
                contrad = self.checker(QUESTION=question, ANSWER1=gold_answer, ANSWER2=gen_answer)
            except Exception as e:
                print(f"-- -- Error generating answer: {e}")
                contrad = None
        
        return dspy.Prediction(
            question=question,
            gold_answer=gold_answer,
            pred_answer=gen_answer if gen_answer else None,
            label=self.process_label(contrad.LABEL) if contrad else None,
            rationale=contrad.RATIONALE if contrad else None,
        )
        
    def process_label(self, label_resp):
        norm = dsp.normalize_text(label_resp).lower()
        if "consistent" in norm:
            clean_pred = "CONSISTENT"
        elif "contradictory" in norm or "contradiction" in norm:
            clean_pred = "CONTRADICTION"
        else:
            clean_pred = "FAILED"
        return clean_pred

###########
# API KEY #
###########
path_env = pathlib.Path(os.getcwd()).parent.parent / '.env'
print(path_env)
load_dotenv(path_env)
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

###########
#   LLM   #
###########
llm = dspy.OpenAI(
    model= "gpt-3.5-turbo",#"gpt-4o-2024-05-13" , #"gpt-4-0125-preview",  #gpt-4o-2024-05-13, #"gpt-4-1106-preview", # TODO: try turbo-instruct,
    max_tokens=1000)
dspy.settings.configure(lm=llm)


############
# DATA #####
############
path_orig_en = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_en_tr.parquet")
path_orig_es = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_es_tr.parquet")
path_source = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/translated_stops_filtered_by_al/df_1.parquet")

path_model = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/LDA_FILTERED_AL/rosie_1_20")
path_corpus_en = path_model / "train_data" / "corpus_EN.txt"
path_corpus_es = path_model / "train_data" / "corpus_ES.txt"

persist_directory = (path_model / 'db_contr_mono').as_posix()

raw = pd.read_parquet(path_source)
with path_corpus_en.open("r", encoding="utf-8") as f:
    lines = [line for line in f.readlines()]
corpus_en = [line.rsplit(" 0 ")[1].strip().split() for line in lines]

ids = [line.split(" 0 ")[0] for line in lines]
df_en = pd.DataFrame({"lemmas": [" ".join(doc) for doc in corpus_en]})
df_en["doc_id"] = ids
df_en["len"] = df_en['lemmas'].apply(lambda x: len(x.split()))
df_en["id_top"] = range(len(df_en))
df_en_raw = df_en.merge(raw, how="inner", on="doc_id")[["doc_id", "id_top", "id_preproc", "lemmas_x", "text", "len"]]

# Read thetas 
thetas = sparse.load_npz(path_model.joinpath(f"mallet_output/{'EN'}/thetas.npz")).toarray()
betas = np.load((path_model.joinpath(f"mallet_output/{'EN'}/betas.npy")))
def get_thetas_str(row,thetas):
    return " ".join([f"{id_}|{round(el, 4)}" for id_,el in enumerate(thetas[row]) if el!=0.0])

def get_most_repr_tpc(row,thetas):
    return np.argmax(thetas[row])

# Save thetas in dataframe and "assigned topic"
df_en_raw["thetas"] = df_en_raw.apply(lambda row: get_thetas_str(row['id_top'], thetas), axis=1)
df_en_raw["id_tpc"] = df_en_raw.apply(lambda row: get_most_repr_tpc(row['id_top'], thetas), axis=1)
tpc = 1
df_tpc = df_en_raw[df_en_raw.id_tpc == tpc]


def create_faiss_index(df, text_column, id_column, model_name="all-mpnet-base-v2", index_file="faiss_index.index"):
    """
    Create a FAISS index from a DataFrame containing text data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    text_column (str): The name of the column containing text data.
    id_column (str): The name of the column containing unique identifiers for the texts.
    model_name (str): The name of the SentenceTransformer model to use for embeddings.
    index_file (str): The file path to save the FAISS index.

    Returns:
    index: The FAISS index object.
    model: The SentenceTransformer model used for embeddings.
    ids: List of document IDs.
    texts: List of document texts.
    """
    texts = df[text_column].tolist()
    ids = df[id_column].tolist()

    model = SentenceTransformer(model_name, device="cuda")

    # Calculate embeddings for the texts
    embeddings = model.encode(texts, show_progress_bar=False)

    # Create a FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  

    # Normalize embeddings to unit length and add to index
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save the index to a file
    faiss.write_index(index, index_file)

    return index, model, ids, texts

def retrieve_similar_documents(query_text, model, index, ids, texts, k=5):
    """
    Retrieve the k most similar documents to the query text.

    Parameters:
    query_text (str): The query text.
    model: The SentenceTransformer model used for embeddings.
    index: The FAISS index object.
    ids (list): List of document IDs.
    texts (list): List of document texts.
    k (int): The number of nearest neighbors to retrieve.

    Returns:
    list: A list of dictionaries containing document IDs, distances, and texts of the k most similar documents.
    """
    # Encode the query text
    query_embedding = model.encode([query_text], show_progress_bar=False)
    faiss.normalize_L2(query_embedding)
    
    # Search the index for the k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the corresponding texts and ids
    results = []
    for i in range(k):
        result = {
            "document_id": ids[indices[0][i]],
            "distance": distances[0][i],
            "text": texts[indices[0][i]]
        }
        results.append(result)
    
    return results

print(f"-- -- Generating index...")
index_en, model_en, ids_en, texts_en = create_faiss_index(df_tpc, text_column='text', id_column='doc_id', index_file='faiss_index_en.index')

############################################################################
# DIVIDER
############################################################################
class Divider(dspy.Signature):
    """
    Using the provided context, extract meaningful and short questions and generate accurate responses based on the context.

    Requirements:
    -------------
    - Each question must refer to one atomic fact from the context.
    - Questions must be as specific as possible and always contain a subject, avoiding expressions like 'this condition', 'this stage'. If asking for a quantity, specify the period of time.
    - Answers should be concise and should not repeat the question.
    - Avoid questions that ask for multiple reasons or complex explanations.

    Examples:
    --------
    CONTEXT: "Nihonium is a synthetic chemical element with symbol Nh and atomic number 113."
    QUESTIONS: {
        "What is the symbol for nihonium?": "Nh",
        "What is the atomic number of nihonium?": "113",
        "Is nihonium naturally occurring or synthetic?": "Synthetic"
    }
    """
    
    CONTEXT = dspy.InputField(desc="Context to generate questions from")
    QUESTIONS = dspy.OutputField(desc="JSON with the questions extracted from the context and their reponses in the format question:answer.")    
    
class DividerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.divider = dspy.ChainOfThought(Divider)
    
    def forward(self, context):
        prediction = self.divider(CONTEXT=context)
        return dspy.Prediction(q=prediction.QUESTIONS)
    
    def parse_response(self, pred):
        try:
            q_cleaned = pred.q.replace('\n', '').replace('    ', '')
            return json.loads(q_cleaned)
        except Exception as e:
            print(f"-- -- Exception occurred: {e}")
            return None

divider = DividerModule()
#output = divider(context="chorioamnionitis: A condition during pregnancy that can cause unexplained fever with uterine tenderness, a high white blood cell count, rapid heart rate in the fetus, rapid heart rate in the woman, and/or foul-smelling vaginal discharge")
#llm.inspect_history(1)

############################################################################
# QA ANSWERER
############################################################################
class QAAnswerer(dspy.Signature):
    """
    Answer the question based only on the given context. If the context does not provide enough information to answer the question, return 'I can't answer that question given the context'.
    """
    
    CONTEXT = dspy.InputField(desc="The context from which the answer should be derived.")
    QUESTION = dspy.InputField(desc="The question to be answered based on the context.")
    ANSWER = dspy.OutputField(desc="The answer to the question, or the message 'I can't answer that questiclearon given the context' indicating insufficient context.")

class QAAnswererModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answerer = dspy.ChainOfThought(QAAnswerer)
    
    def forward(self, context, question):
        try:
            gen_answer = self.answerer(CONTEXT=context, QUESTION=question)
        except Exception as e:
            print(f"-- -- Error generating question: {e}")
            gen_answer = None
        
        return dspy.Prediction(
            gen_answer=gen_answer.ANSWER if gen_answer else None,
        )

k = 5  
responses = []     
checker = LinQAChecker() 
checker.load("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/checker/LinQAChecker-saved.json")
answerer = QAAnswererModule()
for index, row in df_tpc.sample(n=300, random_state=1).iterrows():
    
    print(f"-- -- CHECKING FOR DOC: {row.text}")
    
    # Generate questions
    output = divider(context=row.text)
    
    # Parse questions
    questions = divider.parse_response(output)
    
    if questions:
    
        print(f"-- -- GENERATED QUESTIONS: {questions}")
        
        # Find closest context to the each question
        for qu in questions.keys():
            
            similar_docs = retrieve_similar_documents(qu, model_en, index_en, ids_en, texts_en, k)
            
            similar_docs_ids = [doc["document_id"] for doc in similar_docs if doc["distance"] > 0.6 and doc["document_id"] != row["doc_id"]]
            similar_docs_texts = " || ".join([doc["text"] for doc in similar_docs if doc["distance"] > 0.6 and doc["document_id"] != row["doc_id"]])
            similar_docs_distances = [doc["distance"] for doc in similar_docs if doc["distance"] > 0.6 and doc["document_id"] != row["doc_id"]]
            
            print(f"-- -- CURRENT DOCs: {similar_docs_texts}")
            
            # Generate gen answer
            gen_answer = answerer(context=similar_docs_texts, question=qu).gen_answer
            
            if gen_answer != "I can't answer that question given the context.":
                out = checker(gen_answer=gen_answer, gold_answer=questions[qu], question=qu)
                
                responses.append(
                    [
                        row["doc_id"],
                        similar_docs_ids,
                        row["text"],
                        similar_docs_texts,
                        qu,
                        questions[qu],
                        gen_answer,
                        out["label"],
                        out["rationale"],
                        similar_docs_distances
                    ]
                )
    else:
        print(f"-- -- This was the output: {output}")

results_df = pd.DataFrame(responses,
                          columns=["doc_id1", "doc_id2", "text1", "text2", "q_from_text1", "answer1", "answer2", "label", "rationale", "sim"])

results_df.to_excel("test1_300_k5.xlsx", index=False)
import pdb; pdb.set_trace()
            
