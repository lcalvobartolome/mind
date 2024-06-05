"""
BLADE: Bad Low Affiliation Document Examiner

BLADE is an active learning classifier designed to identify and flag "bad" documents within a corpus. A document is considered "bad" if its words are assigned to all topics with consistently low probabilities, indicating poor topic representation and a lack of strong affiliation with any particular topic. 

Author: Lorena Calvo-Bartolomé
Date: 24.05.2025
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from scipy.sparse import vstack, csr_matrix, hstack
import copy


class Blade(object):
    def __init__(
        self,
        thetas: np.ndarray,
        S3: np.ndarray,
        df: pd.DataFrame,
        logger: logging.Logger = None
    ):
        """
        Initialize the BLADE classifier.

        Parameters
        ----------
        thetas: np.ndarray
            Document-topic distribution
        S3: np.ndarray
            For each document and topic, sum of the betas of the words in the document
        df: pd.DataFrame
            DataFrame containing document metadata
        logger: logging.Logger, optional
            Logger for logging information, by default None
        """
        self._logger = logger if logger else logging.getLogger(__name__)

        # Save input data
        self.thetas = thetas
        self.S3 = S3
        self.df_docs = df
        # Construct features as the concatenation of thetas and S3
        self.X = hstack(
            [csr_matrix(copy.deepcopy(thetas)).astype(np.float64),
             csr_matrix(copy.deepcopy(S3)).astype(np.float64)
             ], format='csr'
        )

        # Read keywords
        try:
            with open("words.txt", "r") as file:
                self.keys = [line.strip() for line in file]
        except:
            self.keys = [
                'infant',
                'postpartum',
                'pregnant',
            ]

        # Initialize classifier
        self._init_classifier()

        # Placeholder for training data
        self.X_train = np.empty((0, self.X.shape[1]))
        self.y_train = np.array([])

        # Initialize pool
        self.X_pool = self.X
        self.df_pool = df.copy()

        # Preprocess positive and negative indices
        self._preprocess_indices()

    def _init_classifier(self):
        """
        Initialize the active learner classifier.
        """
        self.learner = SGDClassifier(loss="log_loss", penalty='l2', tol=1e-3, random_state=42, learning_rate="optimal", eta0=0.1, validation_fraction=0.2, alpha=0.000005)
        self._logger.info("-- -- Active Learner initialized.")

    def _preprocess_indices(self):
        """
        Preprocess indices to identify probable positive and negative samples.
        """
        # Identify probable positive samples
        self.positive_indices = self.df_pool[self.df_pool['text'].str.contains(
            '|'.join(self.keys), case=False, na=False)].index.to_list()

        # Identify probable negative samples
        self.avg_S3 = np.mean(self.S3, axis=1)
        self.negative_indices = np.argsort(self.avg_S3).tolist()

    def update_indices(self, used_index):
        """
        Update the indices by removing the used index from positive and negative indices.

        Parameters
        ----------
        used_index: int
            Index of the used document
        """
        if used_index in self.positive_indices:
            self.positive_indices.remove(used_index)
        if used_index in self.negative_indices:
            self.negative_indices.remove(used_index)

    def preference_function(self, iteration):
        """
        Determine the preferred indices from the pool based on the iteration.

        Parameters
        ----------
        iteration: int
            Current iteration number

        Returns
        -------
        list
            List containing the index of the preferred document
        """
        if len(self.y_train) > 0:
            probas = self.learner.predict_proba(self.X_pool)
            # Calculate uncertainty as the entropy of the prediction probabilities
            uncertainty = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        else:
            uncertainty = np.ones(self.X_pool.shape[0])

        selection_type = iteration % 3

        if selection_type == 0 and len(self.positive_indices) > 0:
            # Select probable positive samples
            positive_uncertainty = uncertainty[self.positive_indices]
            selected_idx = self.positive_indices[np.argmax(
                positive_uncertainty)]

        elif selection_type == 1 and len(self.negative_indices) > 0:
            # Select probable negative samples
            valid_negative_indices = [
                i for i in self.negative_indices if i < len(uncertainty)]
            # Adding a small value to avoid division by zero
            combined_scores = uncertainty[valid_negative_indices] / \
                (self.avg_S3[valid_negative_indices] + 1e-10)
            selected_idx = valid_negative_indices[np.argmax(combined_scores)]

        else:
            # Select based on uncertainty
            selected_idx = np.argmax(uncertainty)

        return [selected_idx]

    def request_labels(self, query_instances, indices):
        """
        Request labels for the queried instances.

        Parameters
        ----------
        query_instances: list
            List of queried instances
        indices: list
            List of indices corresponding to the queried instances

        Returns
        -------
        np.ndarray
            Array of labels provided by the user
        """
        labels = []
        for query_instance, idx in zip(query_instances, indices):
            doc_id = self.df_pool.iloc[idx]['id_top']
            # Assuming there is a 'text' column
            doc_content = self.df_pool.iloc[idx]['text']
            print(f"Document ID: {doc_id}")
            print(f"Document Content: {doc_content}")
            label = int(
                input("Please provide the label for the queried instance (0 or 1): "))
            labels.append(label)
        return np.array(labels)

    def active_learning_loop(self, n_queries=10):
        """
        Perform the active learning loop.

        Parameters
        ----------
        n_queries: int, optional
            Number of queries to perform, by default 10
        """
        for idx in range(n_queries):
            # Use preference function to get the preferred indices from the pool
            preferred_indices = self.preference_function(idx)

            # Select the most preferred instance from the pool
            query_idx = preferred_indices[0]
            query_instance = self.X_pool[query_idx].reshape(1, -1)
            label = self.request_labels(query_instance, [query_idx])

            # Add the queried instance to the training set
            self.X_train = vstack([self.X_train, query_instance])
            self.y_train = np.append(self.y_train, label)

            # Fit the classifier with the new data
            self.learner.partial_fit(
                self.X_train, self.y_train, classes=np.array([0, 1]))

            # Remove queried instance from the pool
            self.X_pool = vstack(
                [self.X_pool[:query_idx], self.X_pool[query_idx+1:]])
            self.df_pool = self.df_pool.drop(
                self.df_pool.index[query_idx]).reset_index(drop=True)

            # Update indices
            self.update_indices(query_idx)

            # Log the process
            self._logger.info(
                f'Iteration {idx + 1}/{n_queries}, Document ID: {query_idx}')

    def predict(self):
        """
        Predict the labels for the remaining pool of documents.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the predicted labels or None if no labeled data is available
        """
        if len(self.y_train) > 0:
            predictions = self.learner.predict(self.X_pool)
            self.df_pool['predicted_label'] = predictions
            return self.df_pool[['id_top', 'text', 'predicted_label']]
        else:
            self._logger.info('No labeled data available to train the model.')
            return None

    def get_predictions_with_text(self):
        """
        Get predictions along with the corresponding document text.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the document text and predicted labels or None if no labeled data is available
        """
        if len(self.y_train) > 0:
            predictions = self.learner.predict(self.X_pool)
            self.df_pool['predicted_label'] = predictions
            return self.df_pool[['text', 'predicted_label']]
        else:
            self._logger.info('No labeled data available to train the model.')
            return None
