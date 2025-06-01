import streamlit as st
import os
import json
import pandas as pd
import io
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import difflib
import re
from datetime import datetime

# === Init API Clients ===
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Constants ===
SIMILARITY_THRESHOLD = 0.7
TOP_K = 3

# === Manual Mapping and Synonyms ===
manual_mapping = {
    'cust_id': 'customer_id',
    'full_name': 'name',
    'contact_email': 'email',
    'signup_date': 'registration_date',
    'mobile_number': 'phone',
    'shipping_address': 'billing_address',
    'rewards_earned': 'loyalty_points',
}
synonym_dict = {
    'customer_id': ['cust_id', 'customerid', 'customer id', 'client_id', 'clientid'],
    'cust_id': ['customer_id', 'customerid', 'customer id', 'client_id', 'clientid'],
    'name': ['full_name', 'fullname', 'full name', 'contact_name', 'person_name'],
    'full_name': ['name', 'fullname', 'full name', 'contact_name', 'person_name'],
    'email': ['contact_email', 'email_address', 'mail', 'emailid'],
    'contact_email': ['email', 'email_address', 'mail', 'emailid'],
    'registration_date': ['signup_date', 'registrationdate', 'registration date', 'join_date', 'created_at'],
    'signup_date': ['registration_date', 'registrationdate', 'registration date', 'join_date', 'created_at'],
    'phone': ['mobile_number', 'telephone', 'mobile', 'cell', 'contact_number'],
    'mobile_number': ['phone', 'telephone', 'mobile', 'cell', 'contact_number'],
    'billing_address': ['shipping_address', 'address', 'location', 'addr', 'home_address'],
    'shipping_address': ['billing_address', 'address', 'location', 'addr', 'home_address'],
    'loyalty_points': ['rewards_earned', 'points', 'reward points'],
    'rewards_earned': ['loyalty_points', 'points', 'reward points'],
    'subscription_type': ['membership_status', 'subscription', 'membership'],
    'membership_status': ['subscription_type', 'subscription', 'membership'],
    'preferred_language': ['preferred_contact_method', 'language', 'contact_method'],
    'preferred_contact_method': ['preferred_language', 'language', 'contact_method'],
}
strict_types = {'date', 'phone', 'id', 'email', 'amount', 'number'}
common_patterns = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'phone': r'^\+?1?\d{9,15}$',
    'date': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
    'id': r'^[A-Z0-9]{3,}$',
    'amount': r'^\$?\d+(\.\d{2})?$'
}

def get_data_type(value):
    if re.match(common_patterns['email'], str(value)):
        return 'email'
    elif re.match(common_patterns['phone'], str(value)):
        return 'phone'
    elif re.match(common_patterns['date'], str(value)):
        return 'date'
    elif re.match(common_patterns['id'], str(value)):
        return 'id'
    elif re.match(common_patterns['amount'], str(value)):
        return 'amount'
    elif str(value).isdigit():
        return 'number'
    else:
        return 'text'

def are_synonyms(field_a, field_b):
    field_a = field_a.lower().replace('_', ' ').replace('-', ' ')
    field_b = field_b.lower().replace('_', ' ').replace('-', ' ')
    for key, synonyms in synonym_dict.items():
        if field_a == key or field_a in synonyms:
            if field_b == key or field_b in synonyms:
                return True
    return False

def compute_field_similarity(field_a, field_b):
    field_a_norm = field_a.lower().replace('_', ' ').replace('-', ' ')
    field_b_norm = field_b.lower().replace('_', ' ').replace('-', ' ')
    if are_synonyms(field_a, field_b):
        return 1.0
    if field_a_norm == field_b_norm:
        return 1.0
    return difflib.SequenceMatcher(None, field_a_norm, field_b_norm).ratio()

def validate_data(data, fields, types, system_name):
    issues = []
    for i, row in enumerate(data):
        for field in fields:
            value = row.get(field, None)
            if value is None or value == '':
                issues.append({
                    "System": system_name,
                    "Row": i+1,
                    "Field": field,
                    "Issue": f"Missing value for '{field}'"
                })
            else:
                expected_type = types[field]
                actual_type = get_data_type(value)
                if expected_type != actual_type:
                    issues.append({
                        "System": system_name,
                        "Row": i+1,
                        "Field": field,
                        "Issue": f"Type mismatch in '{field}' (expected {expected_type}, got {actual_type})"
                    })
    return issues

def match_fields(source_data, target_schema_fields):
    source_fields = list(source_data[0].keys())
    samples_source = {key: str(source_data[0][key]) for key in source_fields}
    types_source = {key: get_data_type(samples_source[key]) for key in source_fields}
    results = []
    matched_source = set()
    matched_target = set()
    audit_log = []
    for target_field in target_schema_fields:
        # Manual mapping override
        source_field = None
        best_score = -1
        status = ""
        method = ""
        if target_field in manual_mapping.values():
            # Find the source field that maps to this target field
            for k, v in manual_mapping.items():
                if v == target_field and k in source_fields:
                    source_field = k
                    best_score = 1.0
                    status = "✅ Strong Match (Manual)"
                    method = "Manual"
                    matched_source.add(source_field)
                    matched_target.add(target_field)
                    break
        if not source_field:
            # Pinecone candidate search: query using the target field name
            query = f"{target_field}"
            vector = openai.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
            # Only search target schema vectors (id starts with 'target_schema_')
            result = index.query(vector=vector, top_k=TOP_K, include_metadata=True, filter=None)
            best_score = -1
            best_match = None
            method = "AI"
            for match in result["matches"]:
                candidate_field = match["metadata"]["field_name"]
                # Try to find a source field that matches this candidate
                for src in source_fields:
                    field_sim = compute_field_similarity(src, candidate_field)
                    score = 0.7 * field_sim + 0.3 * match["score"]
                    if score > best_score:
                        best_score = score
                        source_field = src
                        best_match = candidate_field
            if best_score < SIMILARITY_THRESHOLD or not source_field:
                source_field = 'No Match'
                status = "❌ No Match"
            else:
                status = (
                    "✅ Strong Match" if best_score >= 0.85 else
                    "🟡 Moderate Match" if best_score >= 0.7 else
                    "❌ Weak/Incorrect"
                )
                matched_source.add(source_field)
                matched_target.add(target_field)
        results.append({
            "Target Field": target_field,
            "Source Field": source_field,
            "Source Sample": samples_source[source_field] if source_field in samples_source else '-',
            "Status": status
        })
        audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "Target Field": target_field,
            "Source Field": source_field,
            "Mapping Method": method,
            "AI Score": best_score if method == "AI" else '-',
            "Status": status,
            "User Decision": None
        })
    # Add unmatched source fields
    unmatched_source = set(source_fields) - matched_source
    for src in unmatched_source:
        results.append({
            "Target Field": 'No Match',
            "Source Field": src,
            "Source Sample": samples_source[src],
            "Status": '❌ No Match'
        })
        audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "Target Field": 'No Match',
            "Source Field": src,
            "Mapping Method": 'None',
            "AI Score": '-',
            "Status": '❌ No Match',
            "User Decision": None
        })
    return results, audit_log, types_source

# === Load Sample Data ===
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

data_a = load_json("system_a_data.json")
data_b = load_json("system_b_data.json")
sample_b = data_b[0]
b_index = {r["contact_email"].lower(): r for r in data_b}

# === Ensure output directory exists ===
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Streamlit UI ===
st.title("AI Enabled Data Migration Template")
st.markdown("""
**Instructions:**
- Click **'Run Pre-Migration Data Validation'** to check for missing values and type mismatches in your data. Download the CSV report for details.
- Click **'🔍 Match Fields'** to generate AI-powered field mapping suggestions between your two systems. All fields from both systems will be shown, including unmatched ones.
- Review and approve or reject each suggested mapping. Then click **'✅ Generate Final Output'** to merge and save the normalized data to the project folder and download.
""")

# === Section 1: Pre-Migration Validation ===
st.header("1. Pre-Migration Data Validation")
if st.button("Run Pre-Migration Data Validation"):
    fields_a = list(data_a[0].keys())
    fields_b = list(data_b[0].keys())
    samples_a = {key: str(data_a[0][key]) for key in fields_a}
    samples_b = {key: str(data_b[0][key]) for key in fields_b}
    types_a = {key: get_data_type(samples_a[key]) for key in fields_a}
    types_b = {key: get_data_type(samples_b[key]) for key in fields_b}
    pre_issues_a = validate_data(data_a, fields_a, types_a, 'System A')
    pre_issues_b = validate_data(data_b, fields_b, types_b, 'System B')
    all_issues = pre_issues_a + pre_issues_b
    if all_issues:
        issues_df = pd.DataFrame(all_issues)
        issues_df.to_csv(os.path.join(OUTPUT_DIR, "pre_migration_issues.csv"), index=False)
        st.session_state["pre_issues"] = (len(all_issues), issues_df)
    else:
        st.session_state["pre_issues"] = (0, None)

if "pre_issues" in st.session_state:
    with st.expander("Pre-Migration Validation Results", expanded=True):
        count, issues_df = st.session_state["pre_issues"]
        if count > 0:
            st.warning(f"{count} pre-migration data validation issues found. Download the CSV report for details.")
            st.download_button("⬇️ Download Pre-Migration Issues CSV", data=issues_df.to_csv(index=False), file_name="pre_migration_issues.csv", mime="text/csv")
        else:
            st.success("No pre-migration data validation issues detected.")

# === Section 2: Field Matching ===
st.header("2. Field Mapping Suggestions & Review")
if st.button("🔍 Match Fields"):
    matches, audit_log, types_a = match_fields(data_a, list(data_b[0].keys()))
    st.session_state["matches"] = matches
    st.session_state["audit_log"] = audit_log

if "matches" in st.session_state:
    with st.expander("Field Mapping Suggestions & Review", expanded=True):
        st.subheader("Step 1: Approve / Reject Suggestions")
        st.markdown(
            "> **Note:** All your approve/reject decisions are tracked in the audit log below. Once you finish reviewing, the final output will be generated based on your choices."
        )
        for i, m in enumerate(st.session_state["matches"]):
            col1, col2, col3 = st.columns([3, 3, 2])
            col1.markdown(f"**Target Field:** `{m['Target Field']}`")
            col2.markdown(f"**Matched in Source:** `{m['Source Field']}`")
            decision = col3.radio("Decision", ["Approve", "Reject"], key=f"match_{i}", index=0 if m["Status"].startswith("✅") else 1)
            m["decision"] = decision
            # Update audit log with user decision
            st.session_state["audit_log"][i]["User Decision"] = decision
        st.download_button("⬇️ Download Audit Log", data=pd.DataFrame(st.session_state["audit_log"]).to_csv(index=False), file_name="audit_log.csv", mime="text/csv")

# === Section 3: Merging & Output ===
st.header("3. Merging, Validation & Output Preview")
if st.button("✅ Generate Final Output"):
    approved = {m["Target Field"]: m["Source Field"] for m in st.session_state["matches"] if m["decision"] == "Approve" and m["Source Field"] != 'No Match'}
    rejected_b = {m["Target Field"] for m in st.session_state["matches"] if m["decision"] == "Reject"}
    rejected_a = {m["Source Field"] for m in st.session_state["matches"] if m["decision"] == "Reject" and m["Source Field"] != "No Match"}

    fields_a = set(data_a[0].keys())
    fields_b = set(data_b[0].keys())
    mapped_a = set(approved.values())
    mapped_b = set(approved.keys())
    unmapped_a = fields_a - mapped_a - rejected_a
    unmapped_b = fields_b - mapped_b - rejected_b
    final_fields = list(mapped_a | rejected_a | rejected_b | unmapped_a | unmapped_b)

    merged_data = []
    for row_a in data_a:
        email = row_a.get("email", "").lower()
        row_b = b_index.get(email, {})
        merged = {}
        for b_field, a_field in approved.items():
            merged[a_field] = row_a.get(a_field) or row_b.get(b_field)
        for field in rejected_a:
            merged[field] = row_a.get(field)
        for field in rejected_b:
            merged[field] = row_b.get(field)
        for field in unmapped_a:
            merged[field] = row_a.get(field)
        for field in unmapped_b:
            merged[field] = row_b.get(field)
        merged_data.append(merged)

    # === Post-Migration Validation ===
    post_issues = validate_data(merged_data, final_fields, {k: get_data_type(merged_data[0][k]) for k in final_fields}, 'Merged Output')
    if post_issues:
        post_issues_df = pd.DataFrame(post_issues)
        post_issues_df.to_csv(os.path.join(OUTPUT_DIR, "post_migration_issues.csv"), index=False)
        st.session_state["post_issues"] = (len(post_issues), post_issues_df)
    else:
        st.session_state["post_issues"] = (0, None)

    # Save output to project folder
    with open(os.path.join(OUTPUT_DIR, "normalized_output.json"), "w") as f:
        json.dump(merged_data, f, indent=2)
    df = pd.DataFrame(merged_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "normalized_output.csv"), index=False)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.session_state["merged_data"] = (df, csv_buffer, merged_data, len(final_fields))

if "post_issues" in st.session_state or "merged_data" in st.session_state:
    with st.expander("Output Validation & Preview", expanded=True):
        if "post_issues" in st.session_state:
            count, post_issues_df = st.session_state["post_issues"]
            if count > 0:
                st.warning(f"{count} post-migration data validation issues found. Download the CSV report for details.")
                st.download_button("⬇️ Download Post-Migration Issues CSV", data=post_issues_df.to_csv(index=False), file_name="post_migration_issues.csv", mime="text/csv")
            else:
                st.success("No post-migration data validation issues detected.")
        if "merged_data" in st.session_state:
            df, csv_buffer, merged_data, num_fields = st.session_state["merged_data"]
            st.subheader("🔎 Preview of Merged Output (first 10 rows)")
            st.info(f"🧾 Final output will have {num_fields} columns.")
            st.dataframe(df.head(10))
            st.success("✅ Merged data ready! Download below:")
            st.download_button("⬇️ Download Final Report - JSON", data=json.dumps(merged_data, indent=2), file_name="normalized_output.json", mime="application/json")
            st.download_button("⬇️ Download Final Report - CSV", data=csv_buffer.getvalue().encode("utf-8"), file_name="normalized_output.csv", mime="text/csv")

# Save audit log in output folder
if "audit_log" in st.session_state:
    audit_df = pd.DataFrame(st.session_state["audit_log"])
    audit_df.to_csv(os.path.join(OUTPUT_DIR, "audit_log.csv"), index=False)
