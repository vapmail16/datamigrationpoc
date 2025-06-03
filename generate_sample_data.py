import json
import random
from faker import Faker

fake = Faker()

# Only 5 overlapping fields for System A
OVERLAP_FIELDS = [
    "customer_id", "first_name", "last_name", "email", "date_of_birth"
]

def generate_record_a(i):
    email = fake.email()
    # Introduce random errors for testing
    # 10% chance to make email invalid
    if random.random() < 0.1:
        email = "not_an_email"
    dob = fake.date(pattern="%Y/%m/%d")
    # 10% chance to make dob invalid
    if random.random() < 0.1:
        dob = "not_a_date"
    customer_id = f"A{i:03}"
    # 10% chance to make customer_id a number
    if random.random() < 0.1:
        customer_id = i
    return {
        "customer_id": customer_id,
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "email": email,
        "dob": dob,
    }

NUM_RECORDS = 50

data_a = [generate_record_a(i) for i in range(NUM_RECORDS)]

with open("system_a_data.json", "w") as f:
    json.dump(data_a, f, indent=2)

print("✅ Created system_a_data.json with 50 records and only 5 overlapping fields.")
