# utils/preprocessing.py

import pandas as pd
import os
import pickle

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Fill NaNs with placeholder text
    df.fillna("Unknown", inplace=True)
    
    # Optional: remove ID columns or unhelpful features
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)
    
    return df

def convert_rows_to_chunks(df: pd.DataFrame) -> list:
    chunks = []
    for idx, row in df.iterrows():
        chunk = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(chunk)
    return chunks

def save_chunks(chunks: list, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    # Paths
    CSV_PATH = "data/Training Dataset.csv"
    OUTPUT_PATH = "data/cleaned_chunks.pkl"

    # Step-by-step
    print("📥 Loading and cleaning data...")
    df = load_and_clean_data(CSV_PATH)

    print("🧩 Converting rows to document chunks...")
    chunks = convert_rows_to_chunks(df)

    print("💾 Saving chunks for embedding...")
    save_chunks(chunks, OUTPUT_PATH)
