import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from tqdm import tqdm

def load_json(path):
    """Load JSON data from file."""
    with open(path, "r") as f:
        return json.load(f)

def get_embedding(text, client):
    """Generate OpenAI embedding for text."""
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def build_schema_vectors(schema, client):
    """Create vectors for schema fields with their metadata."""
    vectors = []
    
    for field in schema["fields"]:
        # Create a rich text representation of the field for embedding
        field_text = f"""
        Field Name: {field['name']}
        Data Type: {field['data_type']}
        Required: {field['required']}
        Description: {field['description']}
        Default Value: {field.get('default_value', 'None')}
        """
        
        # Generate embedding
        vector = get_embedding(field_text, client)
        
        # Create metadata dictionary with proper handling of default_value
        metadata = {
            "field_name": field["name"],
            "data_type": field["data_type"],
            "required": field["required"],
            "description": field["description"],
            "schema_version": schema["version"]
        }
        
        # Only add default_value to metadata if it exists and is not None
        if "default_value" in field and field["default_value"] is not None:
            metadata["default_value"] = str(field["default_value"])
        
        # Create vector with metadata
        vectors.append({
            "id": f"target_schema_{field['name']}",
            "values": vector,
            "metadata": metadata
        })
    
    return vectors

def main():
    # Load API keys from .env
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load target schema
    try:
        schema = load_json("schemas/target_schema.json")
        print(f"✅ Loaded target schema: {schema['name']}")
        print(f"Found {len(schema['fields'])} fields")
    except Exception as e:
        print(f"❌ Error loading schema: {str(e)}")
        return
    
    # Create vectors for schema fields
    print("🔁 Creating embeddings for schema fields...")
    vectors = build_schema_vectors(schema, client)
    
    # Upload vectors to Pinecone
    print("📤 Uploading vectors to Pinecone...")
    for i in tqdm(range(0, len(vectors), 10)):
        batch = vectors[i:i+10]
        index.upsert(vectors=batch)
    
    print("✅ Done: Schema metadata uploaded to Pinecone.")
    print("\nUploaded fields:")
    for field in schema["fields"]:
        print(f"- {field['name']} ({field['data_type']})")

if __name__ == "__main__":
    main()
