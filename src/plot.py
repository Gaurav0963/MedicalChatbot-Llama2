import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

class PlotData:
    def __init__(self, predictions, ground_truths, embeddings):
        self.results_dir = PlotData.__create_results_directory()
        self.metrics = PlotData.__evaluate_metrics(predictions, ground_truths, embeddings)

    @staticmethod
    def __create_results_directory(base_folder="results"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = os.path.join(base_folder, f"res-{timestamp}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def __save_plot(fig, filename, save_dir):
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300)  # Increase dpi for publication-quality images
        plt.close(fig)

    # Evaluation Metrics
    @staticmethod
    def __evaluate_metrics(predictions, ground_truths, embeddings):
        """
        Evaluate the chatbot responses using multiple metrics.
        :param predictions: List of chatbot responses.
        :param ground_truths: List of expected correct answers.
        :param embeddings: Embeddings model for similarity comparisons.
        :return: Dictionary with evaluation metrics.
        """
        # Ensure predictions and ground truths have the same length
        assert len(predictions) == len(ground_truths), "Length mismatch between predictions and ground truths."

        # Binary match (1 for correct, 0 for incorrect)
        matches = [1 if pred.strip().lower() == truth.strip().lower() else 0
                for pred, truth in zip(predictions, ground_truths)]

        # Precision, Recall, F1 Score
        precision = precision_score(matches, [1] * len(matches), zero_division=1)
        recall = recall_score(matches, [1] * len(matches), zero_division=1)
        f1 = f1_score(matches, [1] * len(matches), zero_division=1)

        # Accuracy
        accuracy = accuracy_score([1] * len(matches), matches)

        # Cosine Similarity
        pred_embeddings = np.array([embeddings.embed_query(pred) for pred in predictions])
        truth_embeddings = np.array([embeddings.embed_query(truth) for truth in ground_truths])
        cosine_similarities = cosine_similarity(pred_embeddings, truth_embeddings).diagonal()
        avg_cosine_similarity = np.mean(cosine_similarities)

        # ROUGE Scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(pred, truth) for pred, truth in zip(predictions, ground_truths)]
        avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Avg Cosine Similarity": avg_cosine_similarity,
            "Avg ROUGE-1": avg_rouge1,
            "Avg ROUGE-L": avg_rougeL
        } 
        return metrics

    def plot_metrics(self):
        """
        Plot the evaluation metrics for visualization.
        :param metrics: Dictionary containing metric names and values.
        """
        names = list(self.metrics.keys())
        values = list(self.metrics.values())

        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjusted to fit column width
        ax.bar(names, values, width=0.4, color=['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow', 'gray'])
        ax.set_title("Evaluation Metrics", fontsize=12)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xlabel("Metrics", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xticks(names)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        PlotData.__save_plot(fig, "evaluation_metrics.png", self.results_dir)

    def plot_cosine_similarity_distribution(self, cosine_similarities):
        """
        Plot cosine similarity distribution.
        :param cosine_similarities: List of cosine similarity scores.
        """
        sentence_bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        pred_embeddings = sentence_bert_model.encode(self.chatbot_predictions)
        truth_embeddings = sentence_bert_model.encode(self.expected_answers)
        cosine_similarities = cosine_similarity(pred_embeddings, truth_embeddings).diagonal()

        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjusted to fit column width
        sns.histplot(cosine_similarities, bins=20, kde=True, color="skyblue", ax=ax)
        ax.set_title("Cosine Similarity Distribution", fontsize=12)
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        plt.tight_layout()
        PlotData.__save_plot(fig, "cosine_similarity_distribution.png", self.results_dir)

    def plot_rouge_scores(self, rouge_scores):
        """
        Plot ROUGE-1 and ROUGE-L scores for all predictions.
        :param rouge_scores: List of ROUGE scores for each prediction.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(pred, truth) for pred, truth in zip(self.chatbot_predictions, self.expected_answers)]

        rouge1_scores = [score['rouge1'].fmeasure for score in rouge_scores]
        rougeL_scores = [score['rougeL'].fmeasure for score in rouge_scores]

        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjusted to fit column width
        ax.plot(range(len(rouge1_scores)), rouge1_scores, label="ROUGE-1", marker="o", linestyle="--", color="b")
        ax.plot(range(len(rougeL_scores)), rougeL_scores, label="ROUGE-L", marker="s", linestyle="--", color="g")
        ax.set_title("ROUGE Scores", fontsize=12)
        ax.set_xlabel("Prediction Index", fontsize=10)
        ax.set_ylabel("ROUGE Score", fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        PlotData.__save_plot(fig, "rouge_scores.png", self.results_dir)


    def save_metrics_to_json(self): 
        """ 
        Save evaluation metrics to a JSON file. 
        :param metrics: Dictionary containing metric names and values. 
        :param output_directory: Directory where the JSON file will be saved. 
        """ 
        filename = "metrics.json" 
        filepath = os.path.join(self.results_dir, filename) 
        # Save the metrics to a JSON file 
        with open(filepath, 'w') as json_file: 
            json.dump(self.metrics, json_file, indent=4) 