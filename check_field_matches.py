import json
import csv
import difflib
import re
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pandas as pd
from colorama import init, Fore, Style
import numpy as np

# Initialize colorama
init()

class FieldMatcher:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.common_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'date': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
            'id': r'^[A-Z0-9]{3,}$',
            'amount': r'^\$?\d+(\.\d{2})?$'
        }
        # Expanded synonym/mapping dictionary for known field pairs (bi-directional)
        self.synonym_dict = {
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
        # Manual mapping dictionary for explicit overrides (System B field -> System A field)
        self.manual_mapping = {
            'cust_id': 'customer_id',
            'full_name': 'name',
            'contact_email': 'email',
            'signup_date': 'registration_date',
            'mobile_number': 'phone',
            'shipping_address': 'billing_address',
            'rewards_earned': 'loyalty_points',
        }
        # Types that should only match with the same type
        self.strict_types = {'date', 'phone', 'id', 'email', 'amount', 'number'}
        # No match threshold
        self.no_match_threshold = 0.7

    def get_data_type(self, value: str) -> str:
        """Determine the data type of a value."""
        if re.match(self.common_patterns['email'], value):
            return 'email'
        elif re.match(self.common_patterns['phone'], value):
            return 'phone'
        elif re.match(self.common_patterns['date'], value):
            return 'date'
        elif re.match(self.common_patterns['id'], value):
            return 'id'
        elif re.match(self.common_patterns['amount'], value):
            return 'amount'
        elif value.isdigit():
            return 'number'
        else:
            return 'text'

    def are_synonyms(self, field_a: str, field_b: str) -> bool:
        field_a = field_a.lower().replace('_', ' ').replace('-', ' ')
        field_b = field_b.lower().replace('_', ' ').replace('-', ' ')
        for key, synonyms in self.synonym_dict.items():
            if field_a == key or field_a in synonyms:
                if field_b == key or field_b in synonyms:
                    return True
        return False

    def compute_field_similarity(self, field_a: str, field_b: str) -> float:
        """Compute similarity between field names."""
        field_a_norm = field_a.lower().replace('_', ' ').replace('-', ' ')
        field_b_norm = field_b.lower().replace('_', ' ').replace('-', ' ')
        if self.are_synonyms(field_a, field_b):
            return 1.0
        if field_a_norm == field_b_norm:
            return 1.0
        # Use difflib for partial matches
        return difflib.SequenceMatcher(None, field_a_norm, field_b_norm).ratio()

    def compute_sample_similarity(self, sample_a: str, sample_b: str, type_a: str, type_b: str) -> float:
        # For strict types, reduce the weight of sample similarity
        if type_a in self.strict_types and type_b in self.strict_types:
            return 0.0
        embedding_a = self.model.encode(sample_a)
        embedding_b = self.model.encode(sample_b)
        return util.cos_sim(embedding_a, embedding_b).item()

    def compute_type_similarity(self, type_a: str, type_b: str) -> float:
        """Compute similarity between data types."""
        if type_a == type_b:
            return 1.0
        elif (type_a in ['id', 'number'] and type_b in ['id', 'number']) or \
             (type_a in ['text', 'email', 'phone'] and type_b in ['text', 'email', 'phone']):
            return 0.8
        return 0.0

    def match_fields(self, system_a_data: List[Dict], system_b_data: List[Dict]) -> List[Dict]:
        """Match fields between two systems."""
        # Extract fields and samples
        fields_a = list(system_a_data[0].keys())
        fields_b = list(system_b_data[0].keys())
        samples_a = {key: str(system_a_data[0][key]) for key in fields_a}
        samples_b = {key: str(system_b_data[0][key]) for key in fields_b}
        types_a = {key: self.get_data_type(samples_a[key]) for key in fields_a}
        types_b = {key: self.get_data_type(samples_b[key]) for key in fields_b}

        results = []
        matched_a = set()
        matched_b = set()

        for b_field in tqdm(fields_b, desc="Matching fields"):
            best_match = None
            best_score = -1
            best_details = {}
            best_override = False
            b_type = types_b[b_field]

            # Manual mapping override
            if b_field in self.manual_mapping:
                a_field = self.manual_mapping[b_field]
                best_match = a_field
                best_score = 1.0
                best_details = {
                    "field_similarity": 1.0,
                    "sample_similarity": '-',
                    "type_similarity": 1.0
                }
                status = "✅ Strong Match (Manual)"
                matched_a.add(a_field)
                matched_b.add(b_field)
            else:
                for a_field in fields_a:
                    a_type = types_a[a_field]
                    # Type filtering: only allow matches between compatible types
                    if b_type in self.strict_types and a_type in self.strict_types and b_type != a_type:
                        continue
                    # Synonym override
                    if self.are_synonyms(a_field, b_field):
                        field_sim = 1.0
                        type_sim = self.compute_type_similarity(a_type, b_type)
                        score = 0.9 * field_sim + 0.1 * type_sim
                        best_override = True
                        sample_sim = '-'
                    else:
                        field_sim = self.compute_field_similarity(a_field, b_field)
                        sample_sim = self.compute_sample_similarity(samples_a[a_field], samples_b[b_field], a_type, b_type)
                        type_sim = self.compute_type_similarity(a_type, b_type)
                        # Adjusted weights: field name (0.5), type (0.3), sample (0.2)
                        score = 0.5 * field_sim + 0.3 * type_sim + 0.2 * sample_sim
                        # Rule-based boost: if field_sim >= 0.85 and type_sim == 1.0, boost score
                        if field_sim >= 0.85 and type_sim == 1.0:
                            score += 0.1
                    if score > best_score:
                        best_score = score
                        best_match = a_field
                        best_details = {
                            "field_similarity": field_sim,
                            "sample_similarity": sample_sim,
                            "type_similarity": type_sim
                        }
                # No match threshold
                if best_score < self.no_match_threshold:
                    best_match = 'No Match'
                    best_details = {
                        "field_similarity": '-',
                        "sample_similarity": '-',
                        "type_similarity": '-'
                    }
                    status = "❌ No Match"
                else:
                    status = (
                        "✅ Strong Match" if best_score >= 0.85 else
                        "🟡 Moderate Match" if best_score >= 0.7 else
                        "❌ Weak/Incorrect"
                    )
                    matched_a.add(best_match)
                    matched_b.add(b_field)

            results.append({
                "System B Field": b_field,
                "System A Field": best_match,
                "B Sample": samples_b[b_field],
                "A Sample": samples_a[best_match] if best_match in samples_a else '-',
                "Field Similarity": best_details["field_similarity"] if best_details["field_similarity"] == '-' else round(best_details["field_similarity"], 3),
                "Sample Similarity": best_details["sample_similarity"] if best_details["sample_similarity"] == '-' else round(best_details["sample_similarity"], 3),
                "Type Similarity": best_details["type_similarity"] if best_details["type_similarity"] == '-' else round(best_details["type_similarity"], 3),
                "Combined Score": '-' if best_match == 'No Match' else round(best_score, 3),
                "Status": status
            })

        # Add unmatched System A fields
        unmatched_a = set(fields_a) - matched_a
        for a_field in unmatched_a:
            results.append({
                "System B Field": 'No Match',
                "System A Field": a_field,
                "B Sample": '-',
                "A Sample": samples_a[a_field],
                "Field Similarity": '-',
                "Sample Similarity": '-',
                "Type Similarity": '-',
                "Combined Score": '-',
                "Status": '❌ No Match'
            })

        return results

    def display_results_cli(self, results: List[Dict]):
        """Display results in CLI with color coding."""
        print("\n=== Field Matching Results ===\n")
        
        for result in results:
            status_color = (
                Fore.GREEN if "Strong" in result["Status"] else
                Fore.YELLOW if "Moderate" in result["Status"] else
                Fore.RED
            )
            
            print(f"{status_color}{result['Status']}{Style.RESET_ALL}")
            print(f"System B: {result['System B Field']} → System A: {result['System A Field']}")
            print(f"B Sample: {result['B Sample']}")
            print(f"A Sample: {result['A Sample']}")
            print(f"Scores: Field={result['Field Similarity']}, "
                  f"Sample={result['Sample Similarity']}, "
                  f"Type={result['Type Similarity']}, "
                  f"Combined={result['Combined Score']}")
            print("-" * 80)

def main():
    # Load data
    with open("system_a_data.json") as f:
        system_a_data = json.load(f)
    with open("system_b_data.json") as f:
        system_b_data = json.load(f)

    # Initialize matcher and get results
    matcher = FieldMatcher()
    results = matcher.match_fields(system_a_data, system_b_data)

    # Display results in CLI
    matcher.display_results_cli(results)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("field_matches_improved.csv", index=False)
    print(f"\nResults saved to field_matches_improved.csv")

if __name__ == "__main__":
    main()
