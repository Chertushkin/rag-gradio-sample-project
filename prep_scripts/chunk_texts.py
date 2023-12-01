import glob
import os
from typing import List

import numpy as np
import tqdm
import pandas as pd
from semantic_split import SentenceTransformersSimilarity, SimilarSentenceSplitter, SpacySentenceSplitter

model = SentenceTransformersSimilarity(similarity_threshold=0.1)
sentence_splitter = SpacySentenceSplitter()
splitter = SimilarSentenceSplitter(model, sentence_splitter)
TEXTS_DIR = "/home/misha/Coursera/YDS.GenAI/proper/rag-gradio-sample-project/docs_dump"
CHUNKS_DIR = "/home/misha/Coursera/YDS.GenAI/proper/rag-gradio-sample-project/chunks_dump"


def chunkinize_text(text: str) -> List[str]:
    res = splitter.split(text, group_max_sentences=15)
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
