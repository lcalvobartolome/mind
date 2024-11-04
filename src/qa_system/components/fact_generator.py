import logging
import os
import re
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.datasets import Dataset
from typing import Optional
import ast
import contractions
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from dspy.teleprompt import COPRO
from dotenv import load_dotenv
from dspy.evaluate import Evaluate

################################################################################
# DATASET
################################################################################
class FactsDataset(Dataset):

    def __init__(
        self,
        data_fpath: str,
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        text_key: str = "passage",
        seed: Optional[int] = 11235,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._train = []
        self._dev = []
        self._test = []

        # Read the training data
        train_data = pd.read_csv(pathlib.Path(data_fpath))

        train_data, temp_data = train_test_split(
            train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')

################################################################################
# SIGNATURE & MODULE
################################################################################
class GenerateFacts(dspy.Signature):
    """
    Extract self-contained and fully contextualized facts from the given passage.    
    """

    passage = dspy.InputField(
        desc="The passage may contain one or several claims")
    facts = dspy.OutputField(
        desc="List of self-contained and fully contextualized claims in the form 'subject + verb + object' without using pronouns or vague references", prefix="Facts:")

class FactsGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_facts = dspy.Predict(GenerateFacts)

    def process_facts(self, facts):
            # Normalize and clean the facts string
        if "Facts:" in facts:
            facts = facts.split("Facts:", 1)[1]
        elif "facts:" in facts:
            facts = facts.split("facts:", 1)[1]

        try:
            facts = contractions.fix(facts)
        except Exception as e:
            print("Could not expand contractions:", e)

        # Replace problematic characters
        replacements = {
            '’': "'",
            '“': "'",
            '”': "'",
            '"': "'"
        }
        for old_char, new_char in replacements.items():
            facts = facts.replace(old_char, new_char)

        # Handle numbered list format
        if "1." in facts:
            try:
                facts_list = [re.sub(r'^\d+\.\s*', '', fact).strip()
                            for fact in facts.split('\n') if fact.strip()]
                return facts_list
            except Exception as e:
                print("Error processing numbered list:", e)
                return []

        # Handle cases with missing brackets
        facts = facts.strip()
        if facts and not (facts.startswith("[") and facts.endswith("]")):
            facts = facts.strip('[]')  # Remove any stray brackets
            try:
                facts_list = [fact.strip() for fact in facts.split('.') if fact.strip()]
                return facts_list
            except Exception as e:
                print("Error processing facts:", e)
                return []

        # General fallback processing
        try:
            facts_list = [fact.strip() for fact in facts.split('.') if fact.strip()]
        except Exception as e:
            print("General error processing facts:", e)
            return []

        return facts_list

    def forward(self, passage):
        facts = self.generate_facts(passage=passage).facts
        processed_facts = self.process_facts(facts)
        return dspy.Prediction(facts=processed_facts)
    
    
#######################################################################
# FactsGenerator
#######################################################################
class FactsGenerator(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        path_tr_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/tr_data/facts_gpt4_curated.csv",
        trained_promt="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/prompts/FactsGenerator.json",
        do_train=False,
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

        if not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("-- -- Trained prompt not found. Exiting.")
                return
            fg = FactsGeneratorModule()
            fg.load(trained_promt)
            self.module = fg

            self._logger.info(
                f"-- -- FactsGeneratorModule loaded from {trained_promt}")
        else:
            if not path_tr_data:
                self._logger.error(
                    "-- -- Data path is required for training. Exiting.")
                return
            else:
                self._logger.info(
                    f"-- -- Training FactsGeneratorModule from {path_tr_data}")
                self._tr_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.module = self.optimize_module(path_tr_data)
                self.module.save(trained_promt)
                self._logger.info(
                    f"-- -- FactsGeneratorModule trained and saved to {trained_promt}")
                
    def optimize_module(self, data_path, mbd=4, mld=16, ncp=2, mr=2, dev_size=0.25):
        
        dataset = FactsDataset(data_fpath=data_path, dev_size=dev_size)
        
        print(f"-- -- Dataset loaded from {data_path}")
        
        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        print(f"-- -- Dataset split into train, dev, and test. Training module...")
        
        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                    num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.combined_score, **config)
        
        compiled_pred = teleprompter.compile(FactsGeneratorModule(), trainset=trainset, valset=devset)

        #######################################################################
        ## COPRO OPTIMIZATION
        #######################################################################
        #teleprompter = COPRO(
        #    metric=combined_score,
        #    verbose=True,
        #    depth=10
        #    breadth=2,
        #)
        #kwargs = dict(num_threads=64, display_progress=True, display_table=0) 
        #compiled_prompt_opt = teleprompter.compile(FactsGenerator(), trainset=devset,eval_kwargs=kwargs)
        #######################################################################
        
        self._logger.info(f"-- -- Module compiled. Evaluating on test set...")
        
        tests = []
        for el in testset:
            output = compiled_pred(el.passage)
            tests.append([el.passage, el.facts, output["facts"], self.combined_score(el, output)])
            
        df_tests = pd.DataFrame(tests, columns=["passage", "facts", "output", "score"])
        
        self._logger.info(f"-- -- Test set evaluated. Results:")
        self._logger.info(df_tests)

        evaluate = Evaluate(
            devset=devset, metric=self.combined_score, num_threads=1, display_progress=True)
        compiled_score = evaluate(compiled_pred)
        uncompiled_score = evaluate(FactsGeneratorModule())

        print(
            f"## FactsGeneratorModule Score for uncompiled: {uncompiled_score}")
        print(
            f"## FactsGeneratorModule Score for compiled: {compiled_score}")
        print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")

        return compiled_pred
            

    def combined_score(self, example, pred, trace=None):
        def sbert_similarity_score(example, pred, trace=None):
            try:
                scores = []

                predicted_lst = pred["facts"]
                try:
                    gt_lst = ast.literal_eval(example.facts)
                except Exception as e:
                    print("Error in parsing ground truth facts: ", e)
                    gt_lst = example.facts.split(".")

                min_facts = min(len(predicted_lst), len(gt_lst))

                # Generate embeddings for predicted and ground truth facts
                predicted_embeddings = self._tr_model.encode(predicted_lst[:min_facts])
                gt_embeddings = self._tr_model.encode(gt_lst[:min_facts])

                # Calculate cosine similarity for each pair of embeddings
                for pred_emb, gt_emb in zip(predicted_embeddings, gt_embeddings):
                    similarity = 1 - cosine(pred_emb, gt_emb)
                    scores.append(similarity)

                # Return the average similarity score
                return np.mean(scores)

            except Exception as e:
                print("An error occurred: ", e)
                print("predicted_lst: ", predicted_lst)
                print("gt_lst: ", gt_lst)
                return 0.0

        return sbert_similarity_score(example, pred, trace)
    
    def predict(self, passage):
        return self.module(passage=passage).facts