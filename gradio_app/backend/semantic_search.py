import logging
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

# EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
DB_TABLE_NAME = "chunks"

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = AutoModel.from_pretrained(EMB_MODEL_NAME, trust_remote_code=True)
retriever = SentenceTransformer(EMB_MODEL_NAME)

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
table = db.open_table(DB_TABLE_NAME)
