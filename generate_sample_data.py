import json
import random
from faker import Faker

fake = Faker()

# Define similar fields (System A ↔ System B)
common_fields = [
    ("customer_id", "cust_id"),
    ("name", "full_name"),
    ("email", "contact_email"),
    ("registration_date", "signup_date"),
    ("phone", "mobile_number")
]

# Unique fields
unique_fields_A = ["billing_address", "account_balance", "loyalty_points", "subscription_type", "preferred_language"]
unique_fields_B = ["shipping_address", "credit_score", "rewards_earned", "membership_status", "preferred_contact_method"]

# Generate 50 matched email addresses for alignment
matched_emails = [fake.email() for _ in range(50)]

# Helper functions
def generate_common_fields(record_id, email):
    return {
        "customer_id": f"A{record_id:03}",
        "cust_id": f"B{record_id:03}",
        "name": fake.name(),
        "full_name": fake.name(),
        "email": email,
        "contact_email": email,
        "registration_date": fake.date(pattern="%d/%m/%Y"),
        "signup_date": fake.date(pattern="%m-%d-%Y"),
        "phone": fake.phone_number(),
        "mobile_number": fake.phone_number(),
    }

def generate_unique_fields(fields):
    data = {}
    for field in fields:
        if "address" in field:
            data[field] = fake.address().replace("\n", ", ")
        elif "balance" in field:
            data[field] = round(random.uniform(1000, 10000), 2)
        elif "points" in field or "rewards" in field:
            data[field] = random.randint(0, 500)
        elif "subscription" in field or "membership" in field:
            data[field] = random.choice(["Free", "Standard", "Premium"])
        elif "language" in field:
            data[field] = random.choice(["English", "Hindi", "Spanish"])
        elif "contact_method" in field:
            data[field] = random.choice(["Email", "Phone", "SMS"])
        elif "score" in field:
            data[field] = random.randint(300, 850)
        else:
            data[field] = "N/A"
    return data

# Final lists
records_a = []
records_b = []

for i in range(50):
    email = matched_emails[i]

    # Shared fields with different names
    common_data = generate_common_fields(i, email)

    # System A
    record_a = {
        "customer_id": common_data["customer_id"],
        "name": common_data["name"],
        "email": common_data["email"],
        "registration_date": common_data["registration_date"],
        "phone": common_data["phone"]
    }
    record_a.update(generate_unique_fields(unique_fields_A))
    records_a.append(record_a)

    # System B
    record_b = {
        "cust_id": common_data["cust_id"],
        "full_name": common_data["full_name"],
        "contact_email": common_data["contact_email"],
        "signup_date": common_data["signup_date"],
        "mobile_number": common_data["mobile_number"]
    }
    record_b.update(generate_unique_fields(unique_fields_B))
    records_b.append(record_b)

# Save to files
with open("system_a_data.json", "w") as f:
    json.dump(records_a, f, indent=2)

with open("system_b_data.json", "w") as f:
    json.dump(records_b, f, indent=2)

print("✅ Created system_a_data.json and system_b_data.json with 50 records each.")
