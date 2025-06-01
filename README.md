# AI Enabled Data Migration Template

## Overview

This project is an AI-powered data migration and field mapping tool designed to help you map, validate, and merge data from any source system to a defined target schema. It leverages OpenAI embeddings, Pinecone vector search, and rule-based logic to provide accurate, auditable, and human-in-the-loop data migration workflows.

## Features
- **AI-Powered Field Mapping:** Uses OpenAI and Pinecone to suggest field mappings from any source system to a target schema.
- **Manual Mapping & Synonym Support:** Supports manual overrides and synonym dictionaries for robust matching.
- **Human-in-the-Loop Review:** Approve or reject mapping suggestions before merging.
- **Pre/Post-Migration Validation:** Checks for missing values, type mismatches, and anomalies before and after migration.
- **Audit Trail:** Logs all mapping decisions (AI/manual/user) for traceability and compliance.
- **Data Preview:** Preview merged output before downloading.
- **Professional UI:** Streamlit app with clear, persistent sections and downloadable reports.

## Folder Structure
```
project-root/
│
├── match_and_merge_streamlit.py   # Main Streamlit app (source → target schema)
├── ingest_metadata_to_pinecone.py # Ingests target schema metadata into Pinecone
├── define_target_schema.py        # Script to define/edit the target schema
├── check_field_matches.py         # CLI field matching tool
├── generate_sample_data.py        # Sample data generator
├── system_a_data.json             # Example input data (Source System A)
├── system_b_data.json             # Example input data (Source System B)
├── schemas/
│   └── target_schema.json         # The target schema definition
├── output/                        # All generated reports and outputs
│   ├── pre_migration_issues.csv
│   ├── post_migration_issues.csv
│   ├── audit_log.csv
│   ├── normalized_output.json
│   └── normalized_output.csv
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vapmail16/datamigrationpoc.git
   cd datamigrationpoc
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   - Create a `.env` file (do NOT commit this file!) with your API keys:
     ```ini
     OPENAI_API_KEY=your-openai-key
     PINECONE_API_KEY=your-pinecone-key
     PINECONE_INDEX_NAME=your-pinecone-index
     ```
   - **Important:** Ensure `.env` is listed in `.gitignore` before your first commit.

4. **Define and Ingest the Target Schema**
   - Define your target schema:
     ```bash
     python define_target_schema.py
     ```
   - Ingest the target schema into Pinecone:
     ```bash
     python ingest_metadata_to_pinecone.py
     ```

5. **(Optional) Generate Sample Data**
   ```bash
   python generate_sample_data.py
   ```

## Usage

### Run the Streamlit App
```bash
streamlit run match_and_merge_streamlit.py
```

- **Section 1:** Run pre-migration validation and download the issues report if needed.
- **Section 2:** Match source system fields to the target schema, review/approve/reject suggestions, and download the audit log.
- **Section 3:** Generate the final merged output, review post-migration validation, preview the data, and download the final reports.

### Run the CLI Field Matcher (Optional)
```bash
python check_field_matches.py
```

## Output Files
All output files are saved in the `/output` directory:
- `pre_migration_issues.csv` — Pre-migration validation issues
- `post_migration_issues.csv` — Post-migration validation issues
- `audit_log.csv` — All mapping decisions and user actions
- `normalized_output.json` — Final merged data (JSON)
- `normalized_output.csv` — Final merged data (CSV)

## Security Best Practices
- **Never commit your `.env` file or API keys.** Always add `.env` to `.gitignore` before your first commit.
- Always review audit logs and validation reports before sharing data.
- If a secret is accidentally committed, follow [GitHub's guide to removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) and reset your repo history as needed.
- Regenerate any API keys that may have been exposed.

## Contribution Guidelines
- Fork the repository and create a feature branch.
- Submit pull requests with clear descriptions.
- Open issues for bugs, feature requests, or questions.
- Please do not submit API keys or sensitive data in any form.

## License
MIT License

## Contact
For questions or support, open an issue on GitHub or contact the maintainer. 