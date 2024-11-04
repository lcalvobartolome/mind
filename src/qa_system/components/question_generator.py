import logging
import os
import pathlib
from typing import Optional

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

################################################################################
#  DATASET
################################################################################
class QuestionsDataset(Dataset):

    def __init__(
        self,
        data_fpath: str,
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        text_key: str = "fact",
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
class GenerateQuestion(dspy.Signature):
    """Form a close-ended question that directly asks the fact. Question should not be too specific."""
    fact = dspy.InputField()
    question = dspy.OutputField(desc="it asks the fact", prefix="Question:")


class QAGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        #self.generate_question = dspy.Predict(GenerateQuestion)
        self.generate_question = dspy.Predict("fact->question")

    def forward(self, fact):
        # , context=context
        question = self.generate_question(fact=fact).question
        return dspy.Prediction(question=question)

################################################################################
# Question Adapter
################################################################################
class QuestionAdapter(object):
    def __init__(
            self,
            path_tr_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/tr_data/questions_rosie/FullTrialQa7152024.csv",
            logger: logging.Logger = None):
        self._logger = logger or logging.getLogger(__name__)
        self._trf_model = SentenceTransformer('all-MiniLM-L6-v2')

        df_positive = pd.read_csv(path_tr_data)
        df_filtered = df_positive.dropna(
            subset=['question', 'answerPassageText'])
        positive_questions = list(
            set(df_filtered[['question', 'answerPassageText']].question.values.tolist()))

        self.ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.ocsvm.fit(self._trf_model.encode(positive_questions))

    def predict(self, question):
        return self.ocsvm.predict(self._trf_model.encode([question]))[0]

################################################################################
# QAGenerator
################################################################################
class QAGenerator(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        path_tr_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/tr_data/questions_gpt4_curated.csv",
        trained_promt="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/prompts/QAGenerator.json",
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
            qag = QAGeneratorModule()
            qag.load(trained_promt)
            self.module = qag
            self.adapter = QuestionAdapter()

            self._logger.info(
                f"-- -- QAGeneratorModule loaded from {trained_promt}")
        else:
            if not path_tr_data:
                self._logger.error(
                    "-- -- Data path is required for training. Exiting.")
                return
            else:
                self._logger.info(
                    f"-- -- Training QAGeneratorModule from {path_tr_data}")
                self._tr_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.module = self.optimize_module(path_tr_data)
                self.adapter = QuestionAdapter()
                self.module.save(trained_promt)
                self._logger.info(
                    f"-- -- QAGeneratorModule trained and saved to {trained_promt}")

    def optimize_module(self, data_path, mbd=4, mld=16, ncp=2, mr=2, dev_size=0.25):

        dataset = QuestionsDataset(data_fpath=data_path, dev_size=dev_size)

        self._logger.info(f"-- -- Dataset loaded from {data_path}")

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        self._logger.info(
            f"-- -- Dataset split into train, dev, and test. Training module...")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                      num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.combined_score, **config)

        compiled_pred = teleprompter.compile(
            QAGeneratorModule(), trainset=trainset, valset=devset)

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
        # compiled_prompt_opt = teleprompter.compile(QAGeneratorModule(), trainset=devset,eval_kwargs=kwargs)
        #######################################################################

        self._logger.info(f"-- -- Module compiled. Evaluating on test set...")

        tests = []
        for el in testset:
            output = compiled_pred(el.fact)
            tests.append([el.fact, el.question, output["question"],
                         self.combined_score(el, output)])

        df_tests = pd.DataFrame(
            tests, columns=["fact", "ground_q", "pred_q", "score"])

        self._logger.info(f"-- -- Test set evaluated. Results:")
        print(df_tests)

        evaluate = Evaluate(
            devset=devset, metric=self.combined_score, num_threads=1, display_progress=True)
        compiled_score = evaluate(compiled_pred)
        uncompiled_score = evaluate(QAGeneratorModule())

        print(
            f"## QAGeneratorModule Score for uncompiled: {uncompiled_score}")
        print(
            f"## QAGeneratorModule Score for compiled: {compiled_score}")
        print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")

        return compiled_pred

    def combined_score(self, example, pred, trace=None):
        def sbert_similarity_score(example, pred, trace=None):
            try:
                pred_q = pred["question"]
                ground_q = example.question

                # Generate embeddings for predicted and ground truth questions
                pred_q_e = self._tr_model.encode(pred_q)
                ground_q_e = self._tr_model.encode(ground_q)

                return 1 - cosine(pred_q_e, ground_q_e)

            except Exception as e:
                print("An error occurred: ", e)
                print("pred_q: ", pred_q)
                print("ground_q: ", ground_q)
                return 0.0

        return sbert_similarity_score(example, pred, trace)

    def predict(self, fact):
        question = self.module(fact=fact).question
        score = self.adapter.predict(question)
        if score == 1:
            return question, 1
        else:
            return question, 0