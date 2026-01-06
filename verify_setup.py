"""
Quick Test Script - Verify your setup is ready
Run this AFTER installing TensorFlow in a Python 3.11 environment
"""

import sys

print("=" * 70)
print("🔍 FEDERATED LEARNING SETUP VERIFICATION")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
py_version = sys.version_info
if py_version.major == 3 and 9 <= py_version.minor <= 12:
    print("   ✅ Python version is compatible with TensorFlow")
elif py_version.major == 3 and py_version.minor == 14:
    print("   ❌ Python 3.14 detected - TensorFlow NOT supported")
    print("   💡 Solution: Create a new environment with Python 3.11")
    print("      conda create -n fl_project python=3.11")
    print("      conda activate fl_project")
    print("      pip install tensorflow flwr pandas numpy scikit-learn")
else:
    print("   ⚠️  Unusual Python version - TensorFlow compatibility unknown")

# Check required packages
print("\n2. Checking Required Packages:")

packages = {
    'pandas': '✅ Data manipulation',
    'numpy': '✅ Numerical operations',
    'sklearn': '✅ Machine learning tools',
    'tensorflow': '🧠 Deep learning framework',
    'flwr': '🌸 Federated learning',
}

missing_packages = []

for package, description in packages.items():
    try:
        if package == 'sklearn':
            import sklearn
        else:
            __import__(package)
        print(f"   {description} - {package}: Installed")
    except ImportError:
        print(f"   ❌ {package}: NOT FOUND")
        missing_packages.append(package)

# Check datasets
print("\n3. Checking Datasets:")

import os

datasets = {
    '01-12/DrDoS_NTP.csv': 'Client 0 - NTP Attacks',
    '03-11/Portmap.csv': 'Client 1 - Portmap Attacks',
    '01-12/DrDoS_DNS.csv': 'Client 2 - DNS Attacks',
}

base_path = 'C:/Users/nani/Desktop/MINOR'

for dataset, description in datasets.items():
    full_path = f"{base_path}/{dataset}"
    if os.path.exists(full_path):
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"   ✅ {description}: Found ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ {description}: NOT FOUND")
        print(f"      Path: {full_path}")

# Check project files
print("\n4. Checking Project Files:")

project_files = [
    'model.py',
    'client.py',
    'server.py',
    'data_utils.py',
]

for file in project_files:
    if os.path.exists(file):
        print(f"   ✅ {file}: Found")
    else:
        print(f"   ❌ {file}: NOT FOUND")

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\n💡 Installation command:")
    print(f"   pip install {' '.join(missing_packages)}")
else:
    print("\n✅ All packages installed!")

if py_version.major == 3 and 9 <= py_version.minor <= 12:
    print("\n✅ Python version compatible!")
else:
    print("\n❌ Python version incompatible - create Python 3.11 environment!")

print("\n" + "=" * 70)
print("🚀 NEXT STEPS:")
print("=" * 70)

if missing_packages or not (py_version.major == 3 and 9 <= py_version.minor <= 12):
    print("\n1. Fix the issues above")
    print("2. Run this script again")
else:
    print("\n✅ Your setup is ready!")
    print("\nTo start federated learning:")
    print("   Terminal 1: python server.py --rounds 5 --min-clients 3")
    print("   Terminal 2: python client.py 0")
    print("   Terminal 3: python client.py 1")
    print("   Terminal 4: python client.py 2")

print("\n" + "=" * 70 + "\n")
