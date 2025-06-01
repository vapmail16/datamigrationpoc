import os
from dotenv import load_dotenv
from pinecone import Pinecone

def clear_pinecone_index():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        print("❌ Please set PINECONE_API_KEY and PINECONE_INDEX_NAME in your .env file.")
        return

    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes().names()

    if index_name not in indexes:
        print(f"❌ Index '{index_name}' not found.")
        return

    index = pc.Index(index_name)

    # Try listing namespaces to check if index has any data
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        if not namespaces:
            print("ℹ️ Index is already empty. No vectors to delete.")
            return

        print("✅ Namespaces found:", list(namespaces.keys()))
        confirm = input(f"⚠️ Are you sure you want to delete ALL vectors from '{index_name}'? (y/n): ").lower()
        if confirm not in ['y', 'yes']:
            print("Cancelled by user.")
            return

        for ns in namespaces:
            print(f"🧹 Clearing namespace: {ns}")
            index.delete(delete_all=True, namespace=ns)
        print("✅ All vectors deleted.")

    except Exception as e:
        print(f"❌ Error fetching index stats or deleting: {e}")

if __name__ == "__main__":
    clear_pinecone_index()
