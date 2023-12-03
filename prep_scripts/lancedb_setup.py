from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import tqdm
from sentence_transformers import SentenceTransformer

from prep_scripts.constants import (CHUNKS_DIR, DB_TABLE_NAME, DIM,
                                    EMB_MODEL_NAME, TEXT_COLUMN_NAME,
                                    VECTOR_COLUMN_NAME)


def main():
    db = lancedb.connect(".lancedb")  # db location
    batch_size = 1

    model = SentenceTransformer(EMB_MODEL_NAME)
    model.eval()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    schema = pa.schema(
        [pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), DIM)), pa.field(TEXT_COLUMN_NAME, pa.string())]
    )
    tbl = db.create_table(DB_TABLE_NAME, schema=schema, mode="overwrite")

    input_dir = Path(CHUNKS_DIR)
    files = list(input_dir.rglob("*"))

    sentences = []
    for file in files:
        with open(file) as f:
            sentences.append(f.read())

    for i in tqdm.tqdm(range(0, int(np.ceil(len(sentences) / batch_size)))):
        try:
            batch = [sent for sent in sentences[i * batch_size : (i + 1) * batch_size] if len(sent) > 0]
            encoded = model.encode(batch, normalize_embeddings=True, device=device)
            encoded = [list(vec) for vec in encoded]

            df = pd.DataFrame({VECTOR_COLUMN_NAME: encoded, TEXT_COLUMN_NAME: batch})
            tbl.add(df)
        except Exception as e:
            print(e)
            print(f"batch {i} was skipped")

    """
    create ivf-pd index https://lancedb.github.io/lancedb/ann_indexes/
    with the size of the transformer docs, index is not really needed
    but we'll do it for demonstrational purposes
    """
    tbl.create_index(num_partitions=256, num_sub_vectors=96, vector_column_name=VECTOR_COLUMN_NAME)


if __name__ == "__main__":
    main()
