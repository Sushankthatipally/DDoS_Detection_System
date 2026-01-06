import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# CONFIGURATION
# ----------------------------------------------------------------
# Put the path to ONE of your CSV files here.
# Recommendation: Start with 'DrDoS_UDP.csv' from folder 01-12
DATASET_PATH = "C:/Users/nani/Desktop/MINOR/01-12/DrDoS_UDP.csv"# ----------------------------------------------------------------

def load_data(client_id=0, num_clients=3):
    print(f"⏳ Client {client_id}: Loading data from {DATASET_PATH}...")
    
    try:
        # Load a sample of the data (50k rows is enough for a laptop)
        df = pd.read_csv(DATASET_PATH, nrows=50000)
    except FileNotFoundError:
        print(f"❌ Error: File '{DATASET_PATH}' not found.")
        return None, None, None, None

    # 1. Clean Column Names (Remove spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Drop "Junk" Columns that confuse the model
    # We remove IP addresses because we want to detect *patterns*, not specific IPs
    drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Source Port', 'Destination IP', 
                 'Dest IP', 'Destination Port', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 3. Clean Infinite/NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 4. Handle Labels
    # In these CSVs, the column is usually named 'Label'
    if 'Label' in df.columns:
        y = df['Label']
        X = df.drop(columns=['Label'])
    else:
        # Fallback if name is different
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # CHECK: Do we have mixed traffic?
    print(f"ℹ️  Labels found in this file: {y.unique()}")
    
    # 5. Encode Labels (Benign -> 0, Everything else -> 1)
    # This logic forces a binary classification (Attack vs Normal)
    y = y.apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)

    # 6. Partition for Client
    total_samples = len(X)
    samples_per_client = total_samples // num_clients
    start = client_id * samples_per_client
    end = (client_id + 1) * samples_per_client
    
    X_part = X.iloc[start:end]
    y_part = y.iloc[start:end]

    # 7. Normalize
    scaler = StandardScaler()
    X_part = scaler.fit_transform(X_part)

    # 8. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.2, random_state=42)
    
    print(f"✅ Client {client_id}: Ready. {len(X_train)} samples. (Attacks: {sum(y_train==1)}, Normal: {sum(y_train==0)})")
    
    # WARNING if no normal traffic found
    if sum(y_train==0) == 0:
        print("⚠️  WARNING: No 'BENIGN' traffic found in this chunk. The model might fail to learn normal patterns.")
        print("   Solution: Try increasing 'nrows' to 100000 or use a different CSV file.")

    return X_train, X_test, y_train, y_test

def get_input_shape():
    df = pd.read_csv(DATASET_PATH, nrows=1)
    df.columns = df.columns.str.strip()
    drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Source Port', 'Destination IP', 
                 'Dest IP', 'Destination Port', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
    df = df.drop(columns=drop_cols, errors='ignore')
    if 'Label' in df.columns:
        return (df.shape[1] - 1,)
    return (df.shape[1],)

# Test the data loader
if __name__ == "__main__":
    print("🔍 Checking labels in the dataset...")
    X_train, X_test, y_train, y_test = load_data(client_id=0, num_clients=1)
    
    if X_train is not None:
        print(f"\n📊 Dataset Summary:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Attack samples in training: {sum(y_train==1)}")
        print(f"   Benign samples in training: {sum(y_train==0)}")