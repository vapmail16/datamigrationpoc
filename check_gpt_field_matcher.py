import json
import openai
import csv
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_fields_and_samples(data):
    field_samples = {}
    for record in data:
        for k, v in record.items():
            if k not in field_samples:
                field_samples[k] = str(v)
        if len(field_samples) == len(record):
            break
    return field_samples

with open("system_a_data.json") as f:
    system_a_data = json.load(f)

with open("system_b_data.json") as f:
    system_b_data = json.load(f)

system_a_fields = extract_fields_and_samples(system_a_data)
system_b_fields = extract_fields_and_samples(system_b_data)

def ask_gpt(b_field, b_sample, a_field, a_sample):
    prompt = f"""
You are a field matching expert. For each pair of fields and values, determine whether they represent the same concept.

System B Field: "{b_field}" with sample value "{b_sample}"
System A Field: "{a_field}" with sample value "{a_sample}"

Respond in JSON format:
{{
  "match": true or false,
  "confidence": float (0 to 1),
  "reason": "short explanation"
}}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response['choices'][0]['message']['content']
        result = json.loads(content)
        return result.get("match"), result.get("confidence"), result.get("reason")
    except Exception as e:
        return False, 0, f"Error: {str(e)}"

matches = []

for b_field, b_sample in tqdm(system_b_fields.items(), desc="Matching Fields"):
    best_match = None
    highest_conf = 0
    best_reason = ""
    best_a_field = ""
    best_a_sample = ""

    for a_field, a_sample in system_a_fields.items():
        match, conf, reason = ask_gpt(b_field, b_sample, a_field, a_sample)
        if conf > highest_conf:
            highest_conf = conf
            best_match = match
            best_reason = reason
            best_a_field = a_field
            best_a_sample = a_sample

    matches.append({
        "System B Field": b_field,
        "System A Field": best_a_field,
        "B Sample": b_sample,
        "A Sample": best_a_sample,
        "Confidence": highest_conf,
        "Reason": best_reason
    })

with open("gpt_field_matches_output.csv", "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=matches[0].keys())
    writer.writeheader()
    writer.writerows(matches)
