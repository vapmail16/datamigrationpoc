import openai
import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load API key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load JSON files
with open("system_a_data.json") as f:
    system_a_data = json.load(f)

with open("system_b_data.json") as f:
    system_b_data = json.load(f)

# Extract fields and sample values
def get_fields_and_samples(data):
    fields = list(data[0].keys())
    samples = {field: str(data[0].get(field, "")) for field in fields}
    return fields, samples

a_fields, a_samples = get_fields_and_samples(system_a_data)
b_fields, b_samples = get_fields_and_samples(system_b_data)

# Format prompt for GPT
def format_prompt(field_a, sample_a, field_b, sample_b):
    return f"""
Compare the following fields to determine if they represent the same concept.

System A Field: {field_a}
Sample Value: {sample_a}

System B Field: {field_b}
Sample Value: {sample_b}

Do these two fields represent the same concept or information?

Respond strictly in JSON with this format:
{{
  "match": true or false,
  "confidence": float from 0 to 1,
  "reason": "brief explanation"
}}
"""

# GPT wrapper with retry
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def ask_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at identifying equivalent fields in different systems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Compare all fields
results = []
for b_field in tqdm(b_fields, desc="🔍 Matching Fields"):
    for a_field in a_fields:
        sample_b = b_samples.get(b_field, "")
        sample_a = a_samples.get(a_field, "")
        prompt = format_prompt(a_field, sample_a, b_field, sample_b)

        try:
            reply = ask_gpt(prompt)
            parsed = eval(reply[reply.find("{"):reply.rfind("}")+1])
            results.append({
                "System A Field": a_field,
                "System B Field": b_field,
                "A Sample": sample_a,
                "B Sample": sample_b,
                "Match": parsed.get("match"),
                "Confidence": parsed.get("confidence"),
                "Reason": parsed.get("reason"),
            })
        except Exception as e:
            results.append({
                "System A Field": a_field,
                "System B Field": b_field,
                "A Sample": sample_a,
                "B Sample": sample_b,
                "Match": False,
                "Confidence": 0,
                "Reason": f"Error: {str(e)}"
            })

# Output to CSV
df = pd.DataFrame(results)
df.to_csv("gpt_field_matches_output_final.csv", index=False)
print("✅ Output saved to gpt_field_matches_output_final.csv")
