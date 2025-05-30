import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load env vars
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# List indexes
indexes = pc.list_indexes().names()
print("✅ Available indexes:", indexes)
