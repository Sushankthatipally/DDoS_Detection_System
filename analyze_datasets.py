import pandas as pd
import os

folders = ['01-12', '03-11']

print("=" * 80)
print("📊 ANALYZING ALL CSV FILES FOR FEDERATED LEARNING PROJECT")
print("=" * 80)

results = []

for folder in folders:
    folder_path = f"C:/Users/nani/Desktop/MINOR/{folder}"
    if not os.path.exists(folder_path):
        continue
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    print(f"\n{'='*80}")
    print(f"📁 FOLDER: {folder}")
    print(f"{'='*80}")
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            # Read first 50k rows
            df = pd.read_csv(file_path, nrows=50000, low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Find label column
            if 'Label' in df.columns:
                labels = df['Label'].value_counts()
                total = len(df)
                
                print(f"\n📄 {csv_file}")
                print(f"   Total samples: {total:,}")
                print(f"   Labels found:")
                
                benign_count = 0
                attack_count = 0
                
                for label, count in labels.items():
                    percentage = (count / total) * 100
                    print(f"      • {label}: {count:,} ({percentage:.1f}%)")
                    
                    if str(label).strip().upper() == 'BENIGN':
                        benign_count = count
                    else:
                        attack_count += count
                
                # Calculate balance score (closer to 1.0 is better)
                if benign_count > 0 and attack_count > 0:
                    balance_ratio = min(benign_count, attack_count) / max(benign_count, attack_count)
                else:
                    balance_ratio = 0
                
                results.append({
                    'folder': folder,
                    'file': csv_file,
                    'total': total,
                    'benign': benign_count,
                    'attack': attack_count,
                    'balance_score': balance_ratio
                })
                
                # Recommendation
                if balance_ratio > 0.2:
                    print(f"   ✅ GOOD BALANCE (Score: {balance_ratio:.2f})")
                elif benign_count > 0:
                    print(f"   ⚠️  IMBALANCED (Score: {balance_ratio:.2f})")
                else:
                    print(f"   ❌ NO BENIGN DATA")
            else:
                print(f"\n📄 {csv_file}")
                print(f"   ⚠️  No 'Label' column found")
                
        except Exception as e:
            print(f"\n📄 {csv_file}")
            print(f"   ❌ Error: {str(e)[:100]}")

# Final Recommendations
print("\n" + "=" * 80)
print("🎯 RECOMMENDATIONS FOR FEDERATED LEARNING SETUP")
print("=" * 80)

# Sort by balance score
results_sorted = sorted(results, key=lambda x: x['balance_score'], reverse=True)

print("\n📌 BEST FILES (By Balance Score):")
for i, result in enumerate(results_sorted[:5], 1):
    print(f"\n{i}. {result['folder']}/{result['file']}")
    print(f"   Balance Score: {result['balance_score']:.3f}")
    print(f"   Benign: {result['benign']:,} | Attack: {result['attack']:,}")

print("\n" + "=" * 80)
print("💡 SUGGESTED FEDERATED LEARNING SETUP:")
print("=" * 80)

# Find files with good diversity
if len(results_sorted) >= 3:
    print("\n🔹 3-Client Setup (RECOMMENDED):")
    print(f"   Client 1: {results_sorted[0]['folder']}/{results_sorted[0]['file']}")
    print(f"   Client 2: {results_sorted[1]['folder']}/{results_sorted[1]['file']}")
    print(f"   Client 3: {results_sorted[2]['folder']}/{results_sorted[2]['file']}")
    
print("\n✅ Analysis Complete!")
