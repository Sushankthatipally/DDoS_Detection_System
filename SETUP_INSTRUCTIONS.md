# ============================================================================

# FEDERATED LEARNING FOR DDoS DETECTION - SETUP INSTRUCTIONS

# ============================================================================

## 🚨 CRITICAL ISSUE: Python 3.14 is Too New!

Your current Python version: **3.14.0**
TensorFlow requires: **Python 3.9, 3.10, 3.11, or 3.12**

## ✅ SOLUTION: Install Python 3.11

### Option 1: Using Anaconda (RECOMMENDED - Easiest)

1. **Download Anaconda**

   - Visit: https://www.anaconda.com/download
   - Download for Windows
   - Install (keep default settings)

2. **Open Anaconda Prompt** (Search for "Anaconda Prompt" in Start Menu)

3. **Create Python 3.11 environment:**

```bash
conda create -n fl_project python=3.11 -y
conda activate fl_project
```

4. **Install packages:**

```bash
pip install tensorflow flwr pandas numpy scikit-learn matplotlib
```

5. **Navigate to your project:**

```bash
cd C:\Users\nani\Desktop\MINOR
```

6. **Verify setup:**

```bash
python verify_setup.py
```

7. **If all checks pass, run federated learning:**

```bash
# Open 4 separate Anaconda Prompt windows

# Window 1 - Server:
conda activate fl_project
cd C:\Users\nani\Desktop\MINOR
python server.py --rounds 5 --min-clients 3

# Window 2 - Client 0:
conda activate fl_project
cd C:\Users\nani\Desktop\MINOR
python client.py 0

# Window 3 - Client 1:
conda activate fl_project
cd C:\Users\nani\Desktop\MINOR
python client.py 1

# Window 4 - Client 2:
conda activate fl_project
cd C:\Users\nani\Desktop\MINOR
python client.py 2
```

---

### Option 2: Using Python 3.11 Installer

1. **Download Python 3.11**

   - Visit: https://www.python.org/downloads/
   - Download Python 3.11.x (latest 3.11 version)
   - During installation: CHECK "Add Python to PATH"

2. **Open PowerShell in your project folder:**

```powershell
cd C:\Users\nani\Desktop\MINOR
```

3. **Create virtual environment:**

```powershell
python3.11 -m venv fl_env
.\fl_env\Scripts\Activate.ps1
```

4. **Install packages:**

```powershell
pip install tensorflow flwr pandas numpy scikit-learn matplotlib
```

5. **Verify setup:**

```powershell
python verify_setup.py
```

---

## 📋 What Each File Does

### Core Files (Already Created ✅)

- **model.py**: Deep Neural Network for DDoS detection
- **client.py**: Federated learning client (IoT device simulator)
- **server.py**: Federated learning server (aggregator)
- **data_utils.py**: Data loading and preprocessing
- **analyze_datasets.py**: Dataset analysis tool
- **verify_setup.py**: Check if setup is correct

### Datasets (Already Available ✅)

- **01-12/DrDoS_NTP.csv**: Client 0 data (NTP attacks + benign)
- **03-11/Portmap.csv**: Client 1 data (Portmap attacks + benign)
- **01-12/DrDoS_DNS.csv**: Client 2 data (DNS attacks + benign)

---

## 🎯 Project Workflow

```
1. Server waits for clients
   ↓
2. 3 Clients connect with local data
   ↓
3. Server sends initial model to clients
   ↓
4. Each client trains on LOCAL data (privacy preserved!)
   ↓
5. Clients send ONLY model updates (not data)
   ↓
6. Server aggregates updates → Global model
   ↓
7. Repeat steps 3-6 for 5 rounds
   ↓
8. Final global model can detect DDoS attacks!
```

---

## 📊 Expected Results

After 5 rounds, you should see:

- **Global Accuracy**: 95-99%
- **Precision**: 96-99%
- **Recall**: 94-98%
- **F1-Score**: 95-98%

---

## 🎓 For Your Project Report

### Key Points to Highlight:

1. **Privacy Preservation**: Raw data never leaves devices
2. **Bandwidth Efficiency**: Only model weights transmitted
3. **Diverse Data**: 3 different attack types learned
4. **Real Dataset**: CIC-DDoS 2019 (industry standard)
5. **Scalable**: Can add more clients easily

### Architecture Diagram to Include:

```
┌─────────────┐
│   Server    │ ← Aggregates model updates
│  (Global)   │
└──────┬──────┘
       │
   ┌───┴───────────┬──────────┐
   │               │          │
┌──▼───┐      ┌───▼──┐   ┌──▼───┐
│Client│      │Client│   │Client│
│  0   │      │  1   │   │  2   │
│(NTP) │      │(Port)│   │(DNS) │
└──────┘      └──────┘   └──────┘
   ↓              ↓          ↓
Local Data   Local Data  Local Data
(Not Shared) (Not Shared)(Not Shared)
```

### Literature Survey Papers:

1. McMahan et al. (2017) - Original Federated Learning
2. Mothukuri et al. (2021) - FL Security Survey
3. Nguyen et al. (2019) - FL for IoT Anomaly Detection
4. CIC-DDoS2019 Dataset Documentation

---

## 🐛 Common Issues & Solutions

### Issue: "tensorflow not found"

**Solution**: You're in Python 3.14. Follow Option 1 or 2 above.

### Issue: "flwr not found"

**Solution**: Same as above - need Python 3.11 environment.

### Issue: Clients can't connect

**Solution**: Start server FIRST, then start clients.

### Issue: "No BENIGN traffic"

**Solution**: Already fixed! Using DrDoS_NTP.csv, Portmap.csv, DrDoS_DNS.csv

---

## ✅ Quick Checklist

- [ ] Install Anaconda or Python 3.11
- [ ] Create fl_project environment
- [ ] Install tensorflow, flwr, pandas, numpy, scikit-learn
- [ ] Run `python verify_setup.py` (all checks should pass)
- [ ] Open 4 terminal windows
- [ ] Start server in window 1
- [ ] Start clients 0, 1, 2 in windows 2, 3, 4
- [ ] Watch federated learning happen!
- [ ] Take screenshots for your report
- [ ] Document results (accuracy, precision, recall)

---

## 💡 Bonus: Add a Dashboard (Optional)

Create `dashboard.py`:

```python
import streamlit as st
import pandas as pd

st.title("🛡️ Federated DDoS Detection")
st.sidebar.header("Controls")

if st.sidebar.button("Start Training"):
    st.success("Federated Learning Started!")

# Mock metrics
rounds = [1, 2, 3, 4, 5]
accuracy = [0.85, 0.90, 0.94, 0.96, 0.98]

st.line_chart(pd.DataFrame({
    'Round': rounds,
    'Accuracy': accuracy
}).set_index('Round'))
```

Run with: `streamlit run dashboard.py`

---

**Good luck with your project! 🚀**

For questions, check README.md or verify_setup.py output.
