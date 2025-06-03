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
import data_transformation
import logging

logging.basicConfig(level=logging.DEBUG)

# Load target schema fields for output structure (move to top)
with open("schemas/target_schema.json") as f:
    target_schema = json.load(f)
target_fields = [f["name"] for f in target_schema["fields"]]
target_defaults = {f["name"]: f.get("default_value") for f in target_schema["fields"]}

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
    'dob': 'date_of_birth',
    'first_name': 'first_name',
    'last_name': 'last_name',
    'address': 'address',
    'age': 'age',
    'preferences': 'preferences',
    'subscription_tier': 'subscription_tier',
    'last_login': 'last_login',
    'account_balance': 'account_balance',
    'payment_methods': 'payment_methods',
    'notes': 'notes',
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
    'date_of_birth': ['dob', 'dateofbirth', 'date of birth'],
    'dob': ['date_of_birth', 'dateofbirth', 'date of birth'],
    'first_name': ['firstname', 'first name'],
    'last_name': ['lastname', 'last name'],
    'address': ['shipping_address', 'billing_address', 'location', 'addr', 'home_address'],
    'age': ['years', 'years_old'],
    'preferences': ['preference', 'likes', 'interests'],
    'subscription_tier': ['subscription', 'tier', 'plan'],
    'last_login': ['lastlogin', 'last login', 'recent_login'],
    'account_balance': ['balance', 'accountbalance', 'funds'],
    'payment_methods': ['payment', 'methods', 'paymentmethod'],
    'notes': ['note', 'comments', 'remarks'],
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

def match_fields(source_data, target_fields):
    source_fields = list(source_data[0].keys())
    print(f"DEBUG: source_fields = {source_fields}")
    print(f"DEBUG: target_fields = {target_fields}")
    samples_source = {key: str(source_data[0][key]) for key in source_fields}
    types_source = {key: get_data_type(samples_source[key]) for key in source_fields}
    results = []
    matched_target = set()
    audit_log = []
    for target_field in target_fields:
        # Explicit fix: always match date_of_birth with dob if present
        if target_field == "date_of_birth" and "dob" in source_fields:
            print("DEBUG: Explicitly matching date_of_birth with dob")
            source_field = "dob"
            best_score = 1.0
            status = "✅ Strong Match (Manual)"
            method = "Manual"
            matched_target.add(target_field)
            results.append({
                "Target Field": target_field,
                "Source Field": source_field,
                "Source Sample": samples_source[source_field],
                "Status": status
            })
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "Target Field": target_field,
                "Source Field": source_field,
                "Mapping Method": method,
                "AI Score": '-',
                "Status": status,
                "User Decision": None
            })
            continue
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
                    matched_target.add(target_field)
                    break
        if not source_field:
            # Pinecone candidate search: query using the target field name
            query = f"{target_field}"
            if len(target_field) <= 3:
                query += " (date of birth)"
            vector = openai.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
            result = index.query(vector=vector, top_k=TOP_K, include_metadata=True, filter=None)
            best_score = -1
            best_match = None
            method = "AI"
            for match in result["matches"]:
                candidate_field = match["metadata"]["field_name"]
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
    print("DEBUG: Final mapping results:", results)
    return results, audit_log, types_source

# === Load Sample Data ===
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

data_a = load_json("system_a_data.json")
# Remove System B loading and indexing
# sample_b = data_b[0]
# b_index = {r.get("contact_email", r.get("email", "")).lower(): r for r in data_b}

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
    samples_a = {key: str(data_a[0][key]) for key in fields_a}
    types_a = {key: get_data_type(samples_a[key]) for key in fields_a}
    pre_issues_a = validate_data(data_a, fields_a, types_a, 'System A')
    all_issues = pre_issues_a
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
    matches, audit_log, types_a = match_fields(data_a, target_fields)
    st.session_state["matches"] = matches
    st.session_state["audit_log"] = audit_log

if "matches" in st.session_state:
    with st.expander("Field Mapping Suggestions & Review", expanded=True):
        st.subheader("Step 1: Approve / Reject Suggestions")
        st.markdown(
            "> **Note:** All your approve/reject decisions are tracked in the audit log below. Once you finish reviewing, the final output will be generated based on your choices."
        )
        # Only show mappings for real target fields (not 'No Match')
        filtered_matches = [m for m in st.session_state["matches"] if m["Target Field"] != 'No Match']
        for i, m in enumerate(filtered_matches):
            col1, col2, col3 = st.columns([3, 3, 2])
            col1.markdown(f"**Target Field:** `{m['Target Field']}`")
            col2.markdown(f"**Matched in Source:** `{m['Source Field']}`")
            decision = col3.radio("Decision", ["Approve", "Reject"], key=f"match_{i}", index=0 if m["Status"].startswith("✅") else 1)
            m["decision"] = decision
            # Update audit log with user decision
            st.session_state["audit_log"][i]["User Decision"] = decision
        st.download_button("⬇️ Download Audit Log", data=pd.DataFrame(st.session_state["audit_log"]).to_csv(index=False), file_name="audit_log.csv", mime="text/csv")

# === Section 2.5: Transformation Suggestions & Review ===
def get_sample_value(field, data):
    for row in data:
        if field in row and row[field]:
            return row[field]
    return ""

def get_target_sample_value(field, target_schema):
    for f in target_schema["fields"]:
        if f["name"] == field:
            dtype = f.get("data_type", "string")
            fmt = f.get("format")
            if dtype == "date":
                if fmt:
                    return datetime.now().strftime(fmt)
                else:
                    return datetime.now().strftime("%Y-%m-%d")
            elif dtype == "number":
                return "123"
            elif dtype == "boolean":
                return "true"
            elif dtype == "array":
                return "[item1, item2]"
            elif dtype == "object":
                return '{"key": "value"}'
            else:
                return "sample_value"
    return ""

if "matches" in st.session_state:
    # Only consider approved mappings for transformation
    approved_matches = [m for m in st.session_state["matches"] if "decision" in m and m["decision"] == "Approve" and m["Source Field"] != 'No Match']
    if approved_matches:
        st.header("2.5. Transformation Suggestions & Review")
        if "transformations" not in st.session_state:
            st.session_state["transformations"] = {}
        for m in approved_matches:
            src_field = m["Source Field"]
            tgt_field = m["Target Field"]
            src_sample = get_sample_value(src_field, data_a)
            tgt_sample = get_target_sample_value(tgt_field, target_schema)
            # Only get suggestion if not already present
            if tgt_field not in st.session_state["transformations"]:
                with st.spinner(f"Getting transformation suggestion for {src_field} → {tgt_field}..."):
                    suggestion = data_transformation.get_transformation_suggestion(src_field, tgt_field, src_sample, tgt_sample)
                st.session_state["transformations"][tgt_field] = suggestion
            else:
                suggestion = st.session_state["transformations"][tgt_field]
            st.markdown(f"**{src_field} → {tgt_field}**")
            st.markdown(f"Sample Source Value: `{src_sample}`")
            st.markdown(f"Sample Target Value: `{tgt_sample}`")
            st.markdown(f"**AI Suggestion:** {suggestion['description']}")
            # Clean up AI suggestion code (remove markdown/code block formatting)
            def clean_code_block(code):
                if code is None:
                    return ''
                # Remove triple backticks and language hints
                code = re.sub(r'^```[a-zA-Z]*', '', code.strip())
                code = code.replace('```', '').strip()
                # If the code defines a function with a different name, rename it to 'transform'
                # Try to extract the function body if not already in the right format
                if 'def transform(' not in code:
                    match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):([\s\S]*)', code)
                    if match:
                        body = match.group(3)
                        # Remove leading indentation
                        body = '\n'.join(line[4:] if line.startswith('    ') else line for line in body.splitlines())
                        code = f'def transform(x):\n{body if body.strip() else "    return x"}'
                    elif code.strip():
                        # If it's just an expression, wrap it
                        code = f'def transform(x):\n    {code.strip()}'
                    else:
                        code = 'def transform(x):\n    return x'
                # Ensure the function body is indented
                lines = code.splitlines()
                if lines and lines[0].startswith('def transform('):
                    if len(lines) == 1 or not lines[1].startswith('    '):
                        # No body or not indented, add a default indented return
                        code = lines[0] + '\n    return x'
                return code

            # Handle template insertion without session_state error
            template_flag_key = f"insert_template_{tgt_field}"
            if st.button(f"Insert function template for `{tgt_field}`", key=f"template_{tgt_field}"):
                st.session_state[template_flag_key] = True
                st.experimental_rerun()

            # Set the default value for the text area
            default_code = clean_code_block(suggestion['code'])
            # If the field is a date and no valid transform is present, provide a default date transformation
            for f in target_schema["fields"]:
                if f["name"] == tgt_field and f.get("data_type") == "date":
                    # Only override if the default_code is just 'def transform(x):\n    return x' or empty
                    if default_code.strip() in ["def transform(x):\n    return x", "def transform(x):\nreturn x", "", None]:
                        default_code = (
                            "def transform(x):\n"
                            "    from datetime import datetime\n"
                            "    return datetime.strptime(x, '%Y/%m/%d').strftime('%d-%m-%Y')"
                        )
                    break

            code = st.text_area(
                f"Edit transformation code for `{tgt_field}` (must define transform(x))",
                value=default_code,
                key=f"code_{tgt_field}"
            )
            use_transform = st.checkbox(f"Apply transformation for `{tgt_field}`?", value=bool(code and code.lower() != 'none'), key=f"use_{tgt_field}")
            # Save user-edited code and choice
            st.session_state["transformations"][tgt_field]["user_code"] = code
            st.session_state["transformations"][tgt_field]["use_transform"] = use_transform
            if use_transform:
                if not data_transformation.is_valid_transform_code(code):
                    st.warning("⚠️ The transformation code must define a function 'transform(x)'. Please edit the code.")
                else:
                    # Show preview
                    try:
                        preview = data_transformation.safe_apply_transformation(src_sample, code)
                    except Exception as e:
                        preview = f"[Error: {e}]"
                    st.markdown(f"**Preview transformed value:** `{preview}`")
            st.markdown("---")

# === Section 3: Merging & Output ===
st.header("3. Merging, Validation & Output Preview")
if st.button("✅ Generate Final Output"):
    valid_matches = [m for m in st.session_state["matches"] if "decision" in m]
    approved = {m["Target Field"]: m["Source Field"] for m in valid_matches if m["decision"] == "Approve" and m["Source Field"] != 'No Match'}
    rejected_a = {m["Source Field"] for m in valid_matches if m["decision"] == "Reject" and m["Source Field"] != "No Match"}

    fields_a = set(data_a[0].keys())
    mapped_a = set(approved.values())
    final_fields = target_fields

    logging.debug(f"Approved mapping: {approved}")
    merged_data = []
    unmatched_source_fields = list(fields_a - mapped_a - rejected_a)
    unmatched_source_data = []
    for row_idx, row_a in enumerate(data_a):
        merged = {}
        for tgt_field in final_fields:
            a_field = approved.get(tgt_field)
            val = None
            if a_field and a_field in row_a:
                val = row_a.get(a_field)
            if val is None and tgt_field in row_a:
                val = row_a.get(tgt_field)
            # Apply transformation if needed and value is not None
            transform_info = st.session_state.get("transformations", {}).get(tgt_field, {})
            code = transform_info.get("user_code")
            if val is not None and transform_info.get("use_transform") and data_transformation.is_valid_transform_code(code):
                val = data_transformation.safe_apply_transformation(val, code)
            if val is None:
                val = target_defaults.get(tgt_field)
            merged[tgt_field] = val
        merged_data.append(merged)
        unmatched_row = {field: row_a.get(field) for field in unmatched_source_fields}
        unmatched_source_data.append(unmatched_row)

    # === Post-Migration Validation ===
    post_issues = validate_data(merged_data, list(merged_data[0].keys()), {k: get_data_type(merged_data[0][k]) for k in merged_data[0].keys()}, 'Merged Output')
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
    # Save unmatched source data for UI
    st.session_state["unmatched_source_fields"] = (unmatched_source_fields, unmatched_source_data)

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
    # New section: Show unmatched source fields
    if "unmatched_source_fields" in st.session_state:
        unmatched_fields, unmatched_data = st.session_state["unmatched_source_fields"]
        if unmatched_fields:
            st.markdown("---")
            st.subheader(":warning: Unmatched Source Columns (not included in output)")
            st.markdown("These columns from the source data were not mapped to the target schema and are not present in the merged output. Review below:")
            st.dataframe(pd.DataFrame(unmatched_data).head(10))
            st.download_button("⬇️ Download Unmatched Source Columns (CSV)", data=pd.DataFrame(unmatched_data).to_csv(index=False), file_name="unmatched_source_columns.csv", mime="text/csv")

# Save audit log in output folder
if "audit_log" in st.session_state:
    audit_df = pd.DataFrame(st.session_state["audit_log"])
    audit_df.to_csv(os.path.join(OUTPUT_DIR, "audit_log.csv"), index=False)
