import pandas as pd
import torch
import os
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MentalHealthRecommender:
    def __init__(self,model_path,resource_file="resources.csv"):
        # If resource_file is just a filename, force it to look in the same folder as this script
        if not os.path.isabs (resource_file):
            current_dir = os.path.dirname (os.path.abspath (__file__))
            resource_file = os.path.join (current_dir,resource_file)

        self.df = pd.read_csv(resource_file)
        # We load the model as a 'base' model to get embeddings, not classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained (model_path)
        self.model.eval()
        # We pre-calculate the "meaning" vector for every resource description
        resource_texts = (
                self.df["Description"].fillna ("")
                + " "
                + self.df["Tags"].fillna ("")
        ).tolist ()

        self.resource_embeddings = self._batch_encode (resource_texts)


    def _batch_encode(self,texts):
        # Converts a list of text strings into mathematical vectors (Embeddings).
        inputs = self.tokenizer (texts,return_tensors="pt",padding=True,truncation=True,max_length=128)

        with torch.no_grad ():
            outputs = self.model(**inputs)
        # the 'CLS' token (the first token) as the representation of the whole sentence
        embeddings = outputs.last_hidden_state[:,0,:]
        return embeddings.numpy()



    def get_recommendations(self, student_text, self_harm_disclosed=False):

        # Crisis-support resources
        if self_harm_disclosed:
            crisis_resources = self.df[
                self.df["Resource_ID"].isin (
                    ["C001","C002","C003","C004","C005"]
                )
            ].copy ()

            return crisis_resources.to_dict(orient="records")

        # General support recommendations
        safe_df = self.df[
            (self.df["Risk_Level_Allowed"] == 0)
            & (self.df["Resource_ID"] != "S007")
            ].copy ()

        safe_indices = safe_df.index.tolist()
        safe_vectors = self.resource_embeddings[safe_indices]

        student_vector = self._batch_encode([student_text])

        scores = cosine_similarity(
            student_vector,
            safe_vectors
        )[0]

        top_k_indices = scores.argsort()[-3:][::-1]

        recommendations = safe_df.iloc[
            top_k_indices
        ].to_dict(orient="records")

        return recommendations



