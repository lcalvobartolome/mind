import pathlib
import time
import pandas as pd
import numpy as np
from scipy import sparse
from fact_generator_v2 import ClaimsGenerator
from question_generator import QAGenerator
from question_answerer import QAnswerer
from qa_checker import QAChecker
import dspy
from retriever import Index
import pickle

############
# DATA #####
############
path_source = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/multi_blade_filtered_v2/df_1.parquet")
path_model = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/MULTI_BLADE_FILTERED_v2/poly_rosie_1_20")

def get_doc_top_tpcs(doc_distr, topn=2):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    top_weight = [(k, doc_distr[k]) for k in top]
    return top_weight

def get_doc_main_topc(doc_distr):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:1][0]
    return top

raw = pd.read_parquet(path_source)
thetas = sparse.load_npz(path_model / "mallet_output" / "thetas_EN.npz")
raw["thetas"] = list(thetas.toarray())
raw.loc[:, "top_k"] = raw["thetas"].apply(get_doc_top_tpcs)
raw.loc[:, "main_topic"] = raw["thetas"].apply(get_doc_main_topc)
# context is full_doc

# get topic keys in English
with open(path_model / "mallet_output" / "keys_EN.txt", 'r') as file:
    lines = file.readlines()
topic_keys = [line.strip() for line in lines]
for i, key in enumerate(topic_keys):
    print(i, key)

# Separate English and Spanish documents
df_en = raw[raw['doc_id'].str.startswith("EN")].copy()
df_es = raw[raw['doc_id'].str.startswith("ES")].copy()

#  Keep one topic for each corpus
tpc = 12
df_en_tpc = df_en[df_en.main_topic == tpc].copy()
df_es_tpc = df_es[df_es.main_topic == tpc].copy()

# filter English dp_tpc for experiments
#df_en_tpc = df_en_tpc[df_en_tpc['text'].str.contains(
#    'baby', case=False, na=False)]
print(df_en_tpc.shape)
print(df_en_tpc.head())

# Generate Index for Spanish corpus
print("-- -- Generating index for Spanish corpus...")
if pathlib.Path(f"index_store_es_tpc_{tpc}.pkl").exists():
    index = pickle.load(open(f"index_store_es_tpc_{tpc}.pkl", "rb"))
else:
    index = Index(corpus=df_es_tpc.text.values.tolist(),doc_ids=df_es_tpc.doc_id.values.tolist())
    pickle.dump(index, open(f'index_store_es_tpc_{tpc}.pkl', 'wb'))

print("-- -- Index generated.")
print(index.sentences[0:10])

print("Initializing components...")
cg = ClaimsGenerator(do_train=False, model_type="openai", open_ai_model="gpt-4o",path_open_api_key="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/.env")
print("Facts generator initialized.")
qg = QAGenerator(
    do_train=False, model_type="openai", open_ai_model="gpt-4o",
    path_open_api_key="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/.env")
print("Question generator initialized.")
qa = QAnswerer(
    model_type="openai", open_ai_model="gpt-4o",
    path_open_api_key="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/.env", k_similar=10)
print("Question answerer initialized.")
qac = QAChecker(
    do_train=False, model_type="openai", open_ai_model="gpt-4o",
    path_open_api_key="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/.env")
print("QAChecker initialized.")

# Generate claims
print("Generating claims...")

# df_test = pd.read_excel("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/test_tpc12.xlsx")
# claims = df_test.claims.values.tolist()
# questions = df_test.question.values.tolist()
# texts = df_test.text.values.tolist()
# ids = df_test.ids.values.tolist()


class TranslatorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("english->spanish")

    def forward(self, english):
        spanish = self.translate(english=english).spanish
        return spanish
    
class TranslatorModuleToEnglish(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("spanish->english")

    def forward(self, spanish):
        english = self.translate(spanish=spanish).english
        return english


tr = TranslatorModule()
tr_en = TranslatorModuleToEnglish()

results = []

# for c, qu, text, id_ in zip(claims, questions, texts, ids):
#     qu_tr = tr(english=qu)
#     print("Question:", qu)
#     print("Translated question:", qu_tr)
#     answer2, answer2_context, answer2_rationale = \
#         qa.predict(
#             question=qu_tr,
#             context=text, # text_passage
#             model=model_es,
#             index=index_es,
#             ids=ids_es,
#             texts=texts_es
#         )

#     # Check faithfulness of answer2 vs answer1 (i.e., the fact)
#     faithfulness, faithfulness_rationale = qac.predict(
#         answer1=c,
#         answer2=answer2,
#         question=qu
#     )
#     results.append([text, c, qu, answer2, answer2_context, faithfulness, faithfulness_rationale])

for el in df_en_tpc.sample(n=5).iterrows():

    # Generate claims
    claims = cg.predict(el[1].text)
    
    # Generate question for each fact in facts
    for c in claims:
        qu, keep_qu = qg.predict(fact=c)#,context=el[1].text
        
        import pdb; pdb.set_trace()

        if keep_qu == 1:
            # translate question
            qu_tr = tr(english=qu)
            print("Question:", qu)
            print("Translated question:", qu_tr)
            
            import pdb; pdb.set_trace()

            # Answer question
            answer2, answer2_context, answer2_rationale = \
                qa.predict(
                    question=qu_tr,
                    context=el[1].text,
                    index=index
                )
                
            import pdb; pdb.set_trace()

            answer2_tr = tr_en(spanish=answer2)
            # Check faithfulness of answer2 vs answer1 (i.e., the fact)
            faithfulness, faithfulness_rationale = qac.predict(
                answer1=c,
                answer2=answer2_tr,  # answer2,
                question=qu
            )
            import pdb; pdb.set_trace()
            
            results.append([el[1].text, c, qu, qu_tr, answer2, answer2_tr,
                           answer2_context, faithfulness, faithfulness_rationale])

results_df = pd.DataFrame(
    results, columns=[
        "text_passage", "fact", "question", "question_tr",
        "answer", "answer2_tr", "answer2_context", "faithfulness", "faithfulness_rationale"])

results_df.to_excel("results_tpc_19_filtered_baby.xlsx")
