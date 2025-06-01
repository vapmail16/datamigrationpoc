import json
import os
from typing import Dict, List, Optional

class TargetSchema:
    def __init__(self, name: str = "", description: str = ""):
        self.schema = {
            "name": name,
            "description": description,
            "version": "1.0",
            "fields": []
        }
    
    def add_field(self, name: str, data_type: str, required: bool = False, 
                 description: str = "", default_value: Optional[str] = None):
        """Add a field to the schema with its metadata."""
        field = {
            "name": name,
            "data_type": data_type,
            "required": required,
            "description": description,
            "default_value": default_value
        }
        self.schema["fields"].append(field)
    
    def to_json(self) -> Dict:
        """Convert schema to dictionary format."""
        return self.schema
    
    def save_to_file(self, filename: str):
        """Save schema to a JSON file."""
        # Create schemas directory if it doesn't exist
        os.makedirs("schemas", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.schema, f, indent=2)
        print(f"✅ Schema saved to {filename}")

def create_sample_target_schema():
    """Create a sample target schema for demonstration."""
    schema = TargetSchema(
        name="Customer Profile",
        description="Target schema for customer profile data"
    )
    
    # Add some sample fields
    schema.add_field(
        name="customer_id",
        data_type="string",
        required=True,
        description="Unique identifier for the customer"
    )
    
    schema.add_field(
        name="first_name",
        data_type="string",
        required=True,
        description="Customer's first name"
    )
    
    schema.add_field(
        name="last_name",
        data_type="string",
        required=True,
        description="Customer's last name"
    )
    
    schema.add_field(
        name="email",
        data_type="string",
        required=True,
        description="Customer's email address"
    )
    
    schema.add_field(
        name="phone",
        data_type="string",
        required=False,
        description="Customer's phone number"
    )
    
    schema.add_field(
        name="address",
        data_type="object",
        required=False,
        description="Customer's address information"
    )
    
    schema.add_field(
        name="created_at",
        data_type="date",
        required=True,
        description="Date when the customer record was created",
        default_value="current_timestamp"
    )

    schema.add_field(
        name="age",
        data_type="number",
        required=False,
        description="Customer's age in years"
    )

    schema.add_field(
        name="is_active",
        data_type="boolean",
        required=True,
        description="Whether the customer account is active",
        default_value="true"
    )

    schema.add_field(
        name="preferences",
        data_type="array",
        required=False,
        description="List of customer preferences and interests"
    )

    schema.add_field(
        name="subscription_tier",
        data_type="string",
        required=True,
        description="Customer's subscription level (basic, premium, enterprise)",
        default_value="basic"
    )

    schema.add_field(
        name="last_login",
        data_type="date",
        required=False,
        description="Timestamp of customer's last login"
    )

    schema.add_field(
        name="account_balance",
        data_type="number",
        required=True,
        description="Current account balance in USD",
        default_value="0.00"
    )

    schema.add_field(
        name="payment_methods",
        data_type="array",
        required=False,
        description="List of saved payment methods"
    )

    schema.add_field(
        name="notes",
        data_type="string",
        required=False,
        description="Additional notes or comments about the customer"
    )
    
    return schema

def main():
    # Create a sample schema
    schema = create_sample_target_schema()
    
    # Save it to a file
    schema.save_to_file("schemas/target_schema.json")
    
    # Print the schema for verification
    print("\nSchema Contents:")
    print(json.dumps(schema.to_json(), indent=2))

if __name__ == "__main__":
    main() 