import logging
import os
import re
import time
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
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
#  DATASET
################################################################################
class ClaimsDataset(Dataset):

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
        train_data = pd.read_excel(pathlib.Path(data_fpath))
        print("-- -- Length of training data: ", len(train_data))

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
class GenerateClaims(dspy.Signature):
    ("""You are given a passage and you must extract clear, unambiguous claims from the passage without adding additional information. """
     """Ensure each claim is understandable without requiring additional context from the original paragraph. """
     """Avoid vague references like "he," "she," "it," "this". """
     """If a term is mentioned earlier in the passage and needs to be referenced, repeat the full name or description in the claim.""")

    passage = dspy.InputField(
        prefix="Passage:",
        desc="may contain one or several claims"
    )
    claims = dspy.OutputField(
        prefix="Claims:",
        desc="List with the extracted claims"
    )

class ClaimsGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_claims = dspy.ChainOfThought(GenerateClaims)

    def process_claims(self, claims):
        # Normalize and clean the facts string
        if "Claims:" in claims:
            claims = claims.split("Claims:", 1)[1]
        elif "claims:" in claims:
            claims = claims.split("claims:", 1)[1]

        try:
            claims = contractions.fix(claims)
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
            claims = claims.replace(old_char, new_char)

        # Handle numbered list format
        if "1." in claims:
            try:
                claims_list = [re.sub(r'^\d+\.\s*', '', c).strip()
                               for c in claims.split('\n') if c.strip()]
                return claims_list
            except Exception as e:
                print("Error processing numbered list:", e)

        # Handle bullet point format
        try:
            return ast.literal_eval(claims)
        except Exception as e:
            print("Error processing claims with ast:", e)
            print(claims)

        # Handle period separated format
        claims = claims.strip()
        try:
            claims_list = [c.strip() for c in claims.split('.') if c.strip()]
            return claims_list
        except Exception as e:
            print("Error processing claims with split:", e)

        return [claims]

    def forward(self, passage):
        claims = self.generate_claims(passage=passage).claims
        processed_claims = self.process_claims(claims)

        # is_too_short = False
        # if len(processed_claims) == 1 and len(processed_claims[0]) < 1/2 * len(claims):
        #     is_too_short = True

        # vague_references_lst = ["he", "she", "it", "this", "they",
        #                         "them", "their", "those", "these", "that", "there"]
        # is_vague = False
        # for claim in processed_claims:
        #     if any(ref in claim for ref in vague_references_lst):
        #         is_vague = True
        #         break

        # dspy.Assert(
        #     is_too_short, f"There is only one claim extracted from the passage, but there are more claims to be extracted.")
        # dspy.Assert(
        #     is_vague, f"There are vague references in the claims extracted from the passage.")

        return dspy.Prediction(claims=processed_claims)


#######################################################################
# ClaimsGenerator
#######################################################################
class ClaimsGenerator(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        path_tr_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/tr_data/facts.xlsx",
        trained_promt="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/prompts/ClaimsGenerator.json",
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
            cg = ClaimsGeneratorModule()
            cg.load(trained_promt)
            self.module = cg

            self._logger.info(
                f"-- -- ClaimsGeneratorModule loaded from {trained_promt}")
        else:
            if not path_tr_data:
                self._logger.error(
                    "-- -- Data path is required for training. Exiting.")
                return
            else:
                self._logger.info(
                    f"-- -- Training ClaimsGeneratorModule from {path_tr_data}")
                self._tr_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.module = self.optimize_module(path_tr_data)
                self.module.save(trained_promt)
                self._logger.info(
                    f"-- -- ClaimsGeneratorModule trained and saved to {trained_promt}")

    def optimize_module(self, data_path, mbd=4, mld=16, ncp=2, mr=2, dev_size=0.25):

        dataset = ClaimsDataset(data_fpath=data_path, dev_size=dev_size)

        print(f"-- -- Dataset loaded from {data_path}")

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        print(f"-- -- Dataset split into train, dev, and test. Training module...")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld, max_rounds=mr, num_candidate_programs=ncp)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.combined_score, **config)  # WithRandomSearch

        compiled_pred = teleprompter.compile(
            ClaimsGeneratorModule(), trainset=trainset, valset=devset)

        #######################################################################
        # COPRO OPTIMIZATION
        #######################################################################
        # teleprompter = COPRO(
        #    metric=self.combined_score,
        #    verbose=True,
        #    depth=10,
        #    breadth=2,
        # )
        # kwargs = dict(num_threads=64, display_progress=True, display_table=0)
        # compiled_prompt_opt = teleprompter.compile(ClaimsGeneratorModule(), trainset=devset,eval_kwargs=kwargs)
        #######################################################################

        self._logger.info(f"-- -- Module compiled. Evaluating on test set...")

        tests = []
        for el in testset:
            output = compiled_pred(el.passage)
            tests.append([el.passage, el.facts, output["claims"],
                         self.combined_score(el, output)])

        df_tests = pd.DataFrame(
            tests, columns=["passage", "claims", "output", "score"])
        
        print(f"-- -- Test set evaluated. Results:")
        print(df_tests)
        
        evaluate = Evaluate(
            devset=devset, metric=self.combined_score, num_threads=1, display_progress=True)
        compiled_score = evaluate(compiled_pred)
        uncompiled_score = evaluate(ClaimsGeneratorModule())

        print(
            f"## ClaimsGeneratorModule Score for uncompiled: {uncompiled_score}")
        print(
            f"## ClaimsGeneratorModule Score for compiled: {compiled_score}")
        print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")

        return compiled_pred

    def combined_score(self, example, pred, trace=None):
        def sbert_similarity_score(example, pred, trace=None):
            try:
                scores = []

                predicted_lst = pred["claims"]
                try:
                    gt_lst = ast.literal_eval(example.facts)
                except Exception as e:
                    print("Error in parsing ground truth facts: ", e)
                    gt_lst = example.facts.split(".")

                min_facts = min(len(predicted_lst), len(gt_lst))

                # Generate embeddings for predicted and ground truth facts
                predicted_embeddings = self._tr_model.encode(
                    predicted_lst[:min_facts])
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

        def equal_number(example, pred, trace=None):
            try:
                predicted_lst = pred["claims"]
                try:
                    gt_lst = ast.literal_eval(example.facts)
                except Exception as e:
                    print("Error in parsing ground truth facts: ", e)
                    gt_lst = example.facts.split(".")

                length_difference = abs(len(predicted_lst) - len(gt_lst))

                # If the lengths are the same, score is 1
                if length_difference == 0:
                    return 1

                # Otherwise, the score is inversely proportional to the length difference. The higher the difference, the lower the score
                return 1 / (1 + length_difference)

            except Exception as e:
                print("An error occurred: ", e)
                return 0.0

        return (sbert_similarity_score(example, pred, trace) + equal_number(example, pred, trace)) / 2

    def predict(self, passage):
        
        print(self.lm.inspect_history(1))
    
        return self.module(passage=passage).claims
