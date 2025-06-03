import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_transformation_suggestion(source_field, target_field, source_sample, target_sample=None):
    """
    Use OpenAI to suggest a transformation from source_sample to target_sample format.
    Returns a dict with 'description' and (if present) 'code'.
    """
    prompt = f"""
You are a data migration expert.
The source field '{source_field}' has a sample value: '{source_sample}'.
"""
    if target_sample:
        prompt += f"The target field '{target_field}' has a sample value: '{target_sample}'.\n"
    prompt += """
What transformation is needed to convert the source value to the target format? 
If a Python code snippet can do this, provide it.\n
Respond in this format:
Description: <describe the transformation>
Code:
<python code or 'None'>
"""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for data migration."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    content = response.choices[0].message.content
    # Parse response
    desc = ""
    code = None
    if "Description:" in content:
        desc = content.split("Description:",1)[1].split("Code:",1)[0].strip()
    if "Code:" in content:
        code = content.split("Code:",1)[1].strip()
    return {"description": desc, "code": code}

def is_valid_transform_code(code):
    if not code or code.strip().lower() == 'none':
        return False
    return 'def transform(x):' in code

def apply_transformation(value, code):
    """
    Apply a transformation to value using the provided Python code snippet.
    The code should define a function 'transform(x)' and return the result of transform(value).
    """
    if not is_valid_transform_code(code):
        return value
    local_vars = {}
    exec(code, {}, local_vars)
    return local_vars['transform'](value)

def safe_apply_transformation(value, code):
    if not is_valid_transform_code(code):
        return value
    try:
        return apply_transformation(value, code)
    except Exception as e:
        return f"[Transformation Error: {e}]" 