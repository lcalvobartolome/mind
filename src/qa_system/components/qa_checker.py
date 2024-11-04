import logging
import os
import pathlib
from typing import Optional

import dspy
import pandas as pd
from dotenv import load_dotenv
from dspy.datasets import Dataset
from dspy.evaluate import Evaluate
from dspy.teleprompt import COPRO, BootstrapFewShotWithRandomSearch, BootstrapFewShot
from sklearn.model_selection import train_test_split

################################################################################
#  DATASET
################################################################################


class ContradictionsDataset(Dataset):

    def __init__(
        self,
        data_fpath: str,
        type: str,  # "contradictions" or "faithfulness"
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        input_keys: str = ["answer1", "answer2", "question"],
        seed: Optional[int] = 11235,
        *args,
        **kwargs
    ) -> None:
        """
        fact -> question
        """

        super().__init__(*args, **kwargs)

        self._train = []
        self._dev = []
        self._test = []

        # Read the training data
        train_data = pd.read_excel(pathlib.Path(data_fpath))[
            ["answer1", "answer2", "question", "faith_strict", "faithfulness"]]

        for col in train_data.columns:
            train_data[col] = train_data[col].apply(
                lambda x: str(x).strip("\n\t"))

        train_data["faith_strict"] = train_data["faith_strict"].apply(
            lambda x: int(x))
        train_data["faith_strict"] = train_data["faith_strict"].astype(str)

        train_data["faithfulness"] = train_data["faithfulness"].apply(
            lambda x: int(x))
        train_data["faithfulness"] = train_data["faithfulness"].astype(str)

        if type == "contradictions":
            train_data = train_data[
                (train_data.faith_strict == "0") |
                (train_data.faith_strict == "1")
            ]

        train_data, temp_data = train_test_split(
            train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')

################################################################################
# SIGNATURE & MODULE
################################################################################


class CheckAnswersFaithfulness(dspy.Signature):
    """Verify whether ANSWER1 and ANSWER2 are FAITHFUL (1) to each other or not (0) given QUESTION. If its faithfulness can't be determined, return 2."""

    QUESTION = dspy.InputField()
    ANSWER1 = dspy.InputField()
    ANSWER2 = dspy.InputField()
    faithfulness = dspy.OutputField(
        desc="predicted label (1,0, or 2 only)", prefix="Faithfulness:")
    rationale = dspy.OutputField(
        desc="explains the relation between ANSWER1 and ANSWER2", prefix="Rationale:")


class ClassifyContradiction(dspy.Signature):
    ("""Classify the contradiction between ANSWER1 and ANSWER2 given QUESTION into: """
     """(0) Discrepancy: The answers might be correct within their respective contexts, byt they offer conflicting guidance or explanations that could lead to confusion. """
     """(1) Strict: The two answers provide directly opposing information."""
     )

    QUESTION = dspy.InputField()
    ANSWER1 = dspy.InputField()
    ANSWER2 = dspy.InputField()
    contradiction_type = dspy.OutputField(
        desc="predicted label (0 or 1 only)", prefix="Contradiction_type:")
    rationale = dspy.OutputField(
        desc="explains the type of contradiction", prefix="Rationale:")


class QACheckerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.checker = dspy.ChainOfThought(CheckAnswersFaithfulness)

    def process_faithfulness(self, faithfulness):

        try:
            if "0" in faithfulness:
                return 0
            elif "1" in faithfulness:
                return 1
            elif "2" in faithfulness:
                return 2
        except Exception as e:
            print(f"Error: {e}")
            print(f"Faithfulness: {faithfulness}")
            return faithfulness

    def forward(self, answer1, answer2, question):
        response = self.checker(
            ANSWER1=answer1, ANSWER2=answer2, QUESTION=question)
        print(f"-- -- faithfulness: {response.faithfulness}")
        print(f"-- -- rationale: {response.rationale}")

        return dspy.Prediction(faithfulness=self.process_faithfulness(response.faithfulness), rationale=response.rationale)


class ClassifyContradictionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.contrad_classifier = dspy.ChainOfThought(ClassifyContradiction)

    def process_contrad_type(self, contrad_type):
        try:
            if "0" in contrad_type:
                return 0
            elif "1" in contrad_type:
                return 1
        except Exception as e:
            print(f"Error: {e}")
            print(f"Contrd_type: {contrad_type}")
            return contrad_type

    def forward(self, answer1, answer2, question):
        response = self.contrad_classifier(
            ANSWER1=answer1, ANSWER2=answer2, QUESTION=question)
        print(f"-- -- faithfulness: {response.contradiction_type}")
        print(f"-- -- rationale: {response.rationale}")

        return dspy.Prediction(contrad_type=self.process_contrad_type(response.contradiction_type), rationale=response.rationale)

################################################################################
# QAChecker
################################################################################


class QAChecker(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        path_tr_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/tr_data/contradictions_dtset.xlsx",
        trained_promt_checker="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/prompts/QAChecker.json",
        trained_promt_contrad="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/prompts/ClassifyContradiction.json",
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
            if not pathlib.Path(trained_promt_checker).exists():
                self._logger.error(
                    "-- -- Trained prompt for QACheckerModule not found. Exiting.")
                return
            qac = QACheckerModule()
            qac.load(trained_promt_checker)
            self.module1 = qac

            self._logger.info(
                f"-- -- QACheckerModule loaded from {trained_promt_checker}")

            if not pathlib.Path(trained_promt_contrad).exists():
                self._logger.error(
                    "-- -- Trained prompt for ClassifyContradictionModule not found. Exiting.")
                return
            cc = ClassifyContradictionModule()
            cc.load(trained_promt_contrad)
            self.module2 = cc

            self._logger.info(
                f"-- -- ClassifyContradictionModule loaded from {trained_promt_contrad}")

        else:
            if not path_tr_data:
                self._logger.error(
                    "-- -- Data path is required for training. Exiting.")
                return
            else:
                self._logger.info(
                    f"-- -- Training QACheckerModule from {path_tr_data}")
                self.module1 = self.optimize_module(path_tr_data, type="faithfulness")
                self.module1.save(trained_promt_checker)
                self._logger.info(
                    f"-- -- QACheckerModule trained and saved to {trained_promt_checker}")

                self._logger.info(
                    f"-- -- Training ClassifyContradictionModule from {path_tr_data}")
                self.module2 = self.optimize_module(path_tr_data, type="contradictions", mbd=2, mld=1, ncp=1, mr=5)
                self.module2.save(trained_promt_contrad)
                self._logger.info(
                    f"-- -- ClassifyContradictionModule trained and saved to {trained_promt_contrad}")

    def optimize_module(self, data_path, type, mbd=4, mld=16, ncp=2, mr=5, dev_size=0.25):

        model_logs = "QAGeneratorModule" if type == "faithfulness" else "ClassifyContradictionModule"

        dataset = ContradictionsDataset(
            data_fpath=data_path, type=type, dev_size=dev_size)

        self._logger.info(
            f"-- -- Dataset loaded from {data_path} for type: {type}")

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        self._logger.info(
            f"-- -- Dataset split into train, dev, and test. Training module...")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                      num_candidate_programs=ncp, max_rounds=mr)

        if type == "contradictions":
            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=self.combined_score_class, **config)

            compiled_pred = teleprompter.compile(
                ClassifyContradictionModule(), trainset=trainset, valset=devset)

            tests = []
            for el in testset:
                output = compiled_pred(el.answer1, el.answer2, el.question)
                tests.append([el.answer1, el.answer2, el.question, el.faithfulness,
                             output["contrad_type"], output["rationale"], self.combined_score_class(el, output)])

            df_tests = pd.DataFrame(
                tests, columns=["answer1", "answer2", "question", "faithfulness", "pred_faithfulness", "rationale", "score"])

            print(f"-- -- Test set evaluated. Results:")
            print(df_tests)

            evaluate = Evaluate(
                devset=devset, metric=self.combined_score_class, num_threads=1, display_progress=True)
            compiled_score = evaluate(compiled_pred)
            uncompiled_score = evaluate(ClassifyContradictionModule())

        else:

            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=self.combined_score, **config)

            compiled_pred = teleprompter.compile(
                QACheckerModule(), trainset=trainset, valset=devset)

            tests = []
            for el in testset:
                output = compiled_pred(el.answer1, el.answer2, el.question)
                tests.append([el.answer1, el.answer2, el.question, el.faith_strict,
                             output["faithfulness"], output["rationale"], self.combined_score(el, output)])

            df_tests = pd.DataFrame(
                tests, columns=["answer1", "answer2",  "question", "faithfulness", "pred_faithfulness", "rationale", "score"])

            self._logger.info(f"-- -- Test set evaluated. Results:")
            print(df_tests)

            evaluate = Evaluate(
                devset=devset, metric=self.combined_score, num_threads=1, display_progress=True)
            compiled_score = evaluate(compiled_pred)
            uncompiled_score = evaluate(QACheckerModule())

        print(
            f"## {model_logs} Score for uncompiled: {uncompiled_score}")
        print(
            f"## {model_logs} Score for compiled: {compiled_score}")
        print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")
        
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
        # compiled_prompt_opt = teleprompter.compile(QACheckerModule(), trainset=devset,eval_kwargs=kwargs)
        #######################################################################

        return compiled_pred

    def combined_score(self, example, pred, trace=None):

        pred_faith = pred["faithfulness"]
        ground_faith = int(example["faith_strict"]) if isinstance(example["faith_strict"], str) or isinstance(
            example["faith_strict"], float) else example["faith_strict"]

        return 1 if pred_faith == ground_faith else 0

    def combined_score_class(self, example, pred, trace=None):

        pred_faith = pred["contrad_type"]
        ground_faith = int(example["faithfulness"]) if isinstance(example["faithfulness"], str) or isinstance(
            example["faithfulness"], float) else example["faithfulness"]

        return 1 if pred_faith == ground_faith else 0

    def predict(self, answer1, answer2, question):
        pred = self.module1(answer1, answer2, question)
        
        if pred.faithfulness == 1:
            return 2, pred.rationale
        elif pred.faithfulness == 2:
            return 3, pred.rationale
        else:
            pred2 = self.module2(answer1, answer2, question)
            return pred2.contrad_type, (pred.rationale + " | " + pred2.rationale)