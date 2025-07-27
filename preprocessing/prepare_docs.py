# preprocessing/prepare_docs.py

import pandas as pd
import os
import pickle

def row_to_text(row):
    """Convert a row into natural language sentence for document indexing."""
    return (
        f"Applicant {row['Loan_ID']} is a {row['Gender'].lower()} "
        f"{'married' if row['Married'] == 'Yes' else 'unmarried'}, "
        f"{row['Self_Employed'].lower()} individual with {row['Dependents']} dependents. "
        f"Their education level is {row['Education'].lower()} and credit history is "
        f"{'good' if row['Credit_History'] == 1.0 else 'not available or poor'}. "
        f"The applicant is requesting a loan amount of ₹{row['LoanAmount'] * 1000 if pd.notna(row['LoanAmount']) else 'N/A'} "
        f"for a loan term of {row['Loan_Amount_Term']} months. "
        f"The loan is for {row['Property_Area'].lower()} property and the application status was {row['Loan_Status']}."
    )

def prepare_documents(csv_path, save_path):
    df = pd.read_csv(csv_path)

    # Drop rows with too many missing values
    df = df.dropna(subset=["Gender", "Married", "Self_Employed", "Dependents",
                           "Education", "Credit_History", "LoanAmount",
                           "Loan_Amount_Term", "Property_Area", "Loan_Status"])

    documents = [row_to_text(row) for _, row in df.iterrows()]

    # Save documents as pickle file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(documents, f)

    print(f"{len(documents)} documents saved to {save_path}")

if __name__ == "__main__":
    csv_input_path = "./data/Training Dataset.csv"
    output_pickle_path = "./documents/indexed_docs.pkl"
    prepare_documents(csv_input_path, output_pickle_path)
