import logging
import os
import pathlib
from typing import Optional
import faiss

import dspy
import pandas as pd
from dotenv import load_dotenv
from dspy.datasets import Dataset
from dspy.evaluate import Evaluate
from dspy.teleprompt import COPRO, BootstrapFewShotWithRandomSearch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.svm import OneClassSVM
from dsp.utils.utils import deduplicate
from retriever import Index


class QAnswererModule(dspy.Module):
    def __init__(
        self,
        passages_per_hop: int = 2,
        tr_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        k: int = 5
    ):
        super().__init__()

        self.passages_per_hop = passages_per_hop 
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(passages_per_hop)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        self._tr_model = SentenceTransformer(tr_model)
        self._k = k
    
    def forward(self, question, context, index):
        answer_context = []
        
        for hop in range(self.passages_per_hop):
            
            search_query = self.generate_query[hop](context=context, question=question).search_query
                        
            passages = index.retrieve(search_query, topk=self._k)
            
            #retrieve_similar_documents(search_query, context, model, index, ids, texts)
            text_passages = [passage["original_document"] for passage in passages if passage["original_document"] != context]
            
            answer_context = deduplicate(answer_context + text_passages)            
            
        return self.generate_answer(context=answer_context, question=question).copy(context=answer_context, text_passage=context)
    

################################################################################
# QAnswerer
################################################################################
class QAnswerer(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        passages_per_hop: int = 2,
        k_similar : int = 2,
        logger: logging.Logger = None
    ):

        self._logger = logger or logging.getLogger(__name__)

        # Dspy settings
        if model_type == "llama":
            self.lm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B ",
                                       port=8090, url="http://127.0.0.1")
        elif model_type == "openai":
            load_dotenv(path_open_api_key)
            api_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            self.lm = dspy.OpenAI(model=open_ai_model)
        dspy.settings.configure(lm=self.lm)
        
        self.qa = QAnswererModule(passages_per_hop=passages_per_hop, k=k_similar)
        self._logger.info("-- -- QAnswererModule instantiated")
        
    
    def predict(self, question, context, index):
        
        prediction = self.qa(question=question, context=context, index=index)
        
        return prediction.answer, prediction.context, prediction.rationale

    