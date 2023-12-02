import glob
import os
from typing import List
from sentence_transformers import SentenceTransformer, util

import numpy as np
import tqdm
import pandas as pd
from semantic_split import SentenceTransformersSimilarity, SimilarSentenceSplitter, SpacySentenceSplitter

EMB_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"

class SentenceTransformersSimilarity():
    def __init__(self, model='all-MiniLM-L6-v2', similarity_threshold=0.2):
        self.model = SentenceTransformer(model)
        self.similarity_threshold = similarity_threshold


    def similarities(self, sentences: list[str]):
        # Encode all sentences
        embeddings = []
        for sentence in sentences:
            embedding = self.model.encode([sentence])
            embeddings.append(embedding[0])

        # Calculate cosine similarities for neighboring sentences
        similarities = []
        for i in range(1, len(embeddings)):
            sim = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
            similarities.append(sim)

        return similarities

model = SentenceTransformersSimilarity(similarity_threshold=0.15, model=EMB_MODEL_NAME)
sentence_splitter = SpacySentenceSplitter()
splitter = SimilarSentenceSplitter(model, sentence_splitter)
TEXTS_DIR = "/home/misha/Coursera/YDS.GenAI/proper/rag-gradio-sample-project/docs_dump"
CHUNKS_DIR = "/home/misha/Coursera/YDS.GenAI/proper/rag-gradio-sample-project/chunks_dump"


def chunkinize_text(text: str) -> List[str]:
    # return [text]
    res = splitter.split(text, group_max_sentences=50)
    chunks = [" ".join(x) for x in res]
    return chunks


def main():
    text_files = glob.glob(TEXTS_DIR + "/*.txt")
    for text_path in tqdm.tqdm(text_files):
        with open(text_path, "r+") as f:
            text = f.read()
        chunks = chunkinize_text(text)

        output_text_path = text_path.replace("docs_dump", "chunks_dump").replace(".txt", "")
        if not os.path.exists(os.path.dirname(output_text_path)):
            os.makedirs(os.path.dirname(output_text_path))
        for i, chunk in enumerate(chunks):
            output_chunk_path = f"{output_text_path}_chunk_{i}.txt"
            # print(output_chunk_path)
            with open(output_chunk_path, "w+") as f:
                f.write(chunk)


if __name__ == "__main__":
    main()
