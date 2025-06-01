import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# View index stats
stats = index.describe_index_stats()
print("📊 Index Stats:")
print(stats)

# Fetch 2 sample records
sample_ids = ["A_customer_id", "B_cust_id"]
result = index.fetch(ids=sample_ids)
print("🔍 Sample fetched vectors:")
print(result)
