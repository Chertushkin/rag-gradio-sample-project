import glob
import os
from typing import List

import tqdm
from semantic_split import SimilarSentenceSplitter, SpacySentenceSplitter
from sentence_transformers import SentenceTransformer, util

from prep_scripts.constants import EMB_MODEL_NAME, TEXTS_DIR


class CUDAFriendlySentenceTransformersSimilarity:
    def __init__(self, model="all-MiniLM-L6-v2", similarity_threshold=0.2):
        self.model = SentenceTransformer(model)
        self.similarity_threshold = similarity_threshold

    def similarities(self, sentences: list[str]):
        embeddings = []
        for sentence in sentences:
            embedding = self.model.encode([sentence])
            embeddings.append(embedding[0])

        similarities = []
        for i in range(1, len(embeddings)):
            sim = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()
            similarities.append(sim)

        return similarities

def chunkinize_text(text: str, splitter) -> List[str]:
    # return [text]
    res = splitter.split(text, group_max_sentences=20)
    chunks = [" ".join(x) for x in res]
    return chunks


def main():
    similarity_model = CUDAFriendlySentenceTransformersSimilarity(similarity_threshold=0.15, model=EMB_MODEL_NAME)
    sentence_splitter = SpacySentenceSplitter()
    splitter = SimilarSentenceSplitter(similarity_model, sentence_splitter)

    text_files = glob.glob(TEXTS_DIR + "/*.txt")
    for text_path in tqdm.tqdm(text_files):
        with open(text_path, "r+") as f:
            text = f.read()
        chunks = chunkinize_text(text, splitter)

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
