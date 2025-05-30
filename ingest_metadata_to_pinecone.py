import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from tqdm import tqdm

# Load API keys from .env
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load JSON data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

data_a = load_json("system_a_data.json")
data_b = load_json("system_b_data.json")
sample_a = data_a[0]
sample_b = data_b[0]

# Generate OpenAI embeddings
def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Create vectors for Pinecone
def build_vectors(system_name, record):
    vectors = []
    for field, value in record.items():
        combined = f"{field}: {value}"
        vector = embed_text(combined)
        vectors.append({
            "id": f"{system_name}_{field}",
            "values": vector,
            "metadata": {
                "system": system_name,
                "field": field,
                "sample": str(value)
            }
        })
    return vectors

# Ingest both System A and System B fields
print("🔁 Embedding & uploading...")
vectors = build_vectors("A", sample_a) + build_vectors("B", sample_b)

for i in tqdm(range(0, len(vectors), 10)):
    batch = vectors[i:i+10]
    index.upsert(vectors=batch)

print("✅ Done: Embeddings uploaded to Pinecone.")
