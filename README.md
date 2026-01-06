# Federated Learning for DDoS Attack Detection in IoT Networks

## 📋 Project Overview

This project implements a **Federated Learning (FL)** system for detecting DDoS attacks in IoT networks. The system allows multiple IoT devices to collaboratively train a global model **without sharing raw data**, preserving privacy and reducing bandwidth.

## 🎯 Key Features

- ✅ **Privacy-Preserving**: Raw data never leaves the device
- ✅ **Distributed Learning**: 3 IoT clients with different attack types
- ✅ **Real-time Detection**: Binary classification (Attack vs Normal)
- ✅ **CIC-DDoS 2019 Dataset**: Industry-standard network traffic data

## 📁 Project Structure

```
MINOR/
├── model.py           # Deep Neural Network architecture
├── client.py          # Federated client (IoT device simulator)
├── server.py          # Federated server (aggregator)
├── data_utils.py      # Data loading and preprocessing
├── analyze_datasets.py # Dataset analysis tool
├── requirements.txt   # Python dependencies
├── README.md         # This file
├── 01-12/            # Dataset folder 1 (DrDoS attacks)
└── 03-11/            # Dataset folder 2 (Various attacks)
```

## 🚨 **IMPORTANT: Python Version Issue**

**Your current Python version is 3.14**, which is **too new** for TensorFlow!

### ✅ Solution: Create a New Environment with Python 3.11

**Option 1: Using Anaconda (RECOMMENDED)**

```powershell
# Download Python 3.11 Anaconda from: https://www.anaconda.com/download

# Create new environment
conda create -n fl_project python=3.11
conda activate fl_project

# Install packages
pip install tensorflow flwr pandas numpy scikit-learn matplotlib
```

**Option 2: Using Python 3.11 Installer**

```powershell
# Download Python 3.11 from: https://www.python.org/downloads/

# Create virtual environment
python3.11 -m venv fl_env
.\fl_env\Scripts\Activate.ps1

# Install packages
pip install tensorflow flwr pandas numpy scikit-learn matplotlib
```

**Option 3: Wait for TensorFlow to support Python 3.14 (not recommended)**

## 📦 Installation (After fixing Python version)

1. **Activate your Python 3.11 environment**

```powershell
conda activate fl_project  # or .\fl_env\Scripts\Activate.ps1
```

2. **Install dependencies**

```powershell
pip install -r requirements.txt
```

3. **Verify installation**

```powershell
python model.py
```

## 🚀 How to Run Federated Learning

### Step 1: Start the Server

Open **Terminal 1** and run:

```powershell
python server.py --rounds 5 --min-clients 3
```

This starts the federated server that will aggregate model updates from 3 clients.

### Step 2: Start Client 1 (NTP Attack Detection)

Open **Terminal 2** and run:

```powershell
python client.py 0
```

### Step 3: Start Client 2 (Portmap Attack Detection)

Open **Terminal 3** and run:

```powershell
python client.py 1
```

### Step 4: Start Client 3 (DNS Attack Detection)

Open **Terminal 4** and run:

```powershell
python client.py 2
```

### Expected Output

The server will:

1. Wait for all 3 clients to connect
2. Run 5 rounds of federated training
3. Display metrics after each round
4. Show final global model performance

## 📊 Dataset Information

### Selected Datasets (Best Balance)

| Client | Dataset             | Attack Type    | Benign Samples | Attack Samples |
| ------ | ------------------- | -------------- | -------------- | -------------- |
| 0      | 01-12/DrDoS_NTP.csv | NTP Reflection | 12,347 (24.7%) | 37,653 (75.3%) |
| 1      | 03-11/Portmap.csv   | Portmap Attack | 4,717 (9.4%)   | 45,283 (90.6%) |
| 2      | 01-12/DrDoS_DNS.csv | DNS Reflection | 1,702 (3.4%)   | 48,298 (96.6%) |

**Total**: ~18,766 benign samples across all clients

### Why These Files?

- ✅ **Best balance** of attack and normal traffic
- ✅ **Diverse attack types** for robust learning
- ✅ **Realistic scenario** for federated learning

## 🧪 Testing Individual Components

### Test Model

```powershell
python model.py
```

### Test Data Loading

```powershell
python data_utils.py
```

### Analyze All Datasets

```powershell
python analyze_datasets.py
```

## 📈 Expected Results

After 5 rounds of federated learning, you should see:

- **Accuracy**: 95-99% (DDoS attacks have distinct patterns)
- **Precision**: 96-99% (Few false positives)
- **Recall**: 94-98% (Detects most attacks)
- **F1-Score**: 95-98%

## 🎨 UI Design (Future Work)

As noted in your project document, you can add:

### Streamlit Dashboard

```python
# dashboard.py
import streamlit as st

st.title("🛡️ Federated DDoS Detection Dashboard")
st.sidebar.button("Start Server")
st.sidebar.button("Start Clients")
st.line_chart(accuracy_per_round)
st.dataframe(attack_log)
```

Run with: `streamlit run dashboard.py`

## 📚 Literature Survey References

Include these in your report:

1. **McMahan et al. (2017)** - "Communication-Efficient Learning of Deep Networks from Decentralized Data"

   - Original Federated Learning paper from Google
   - https://arxiv.org/abs/1602.05629

2. **Mothukuri et al. (2021)** - "A Survey on Security and Privacy of Federated Learning"

   - Comprehensive survey on FL security
   - IEEE Access

3. **Nguyen et al. (2019)** - "DÏoT: A Federated Self-learning Anomaly Detection System for IoT"

   - FL for IoT anomaly detection
   - IEEE ICDCS

4. **CIC-DDoS2019 Dataset** - Canadian Institute for Cybersecurity
   - https://www.unb.ca/cic/datasets/ddos-2019.html

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**: You're using Python 3.14. Follow the "Python Version Issue" section above.

### Issue: "No matching distribution found for tensorflow"

**Solution**: TensorFlow only supports Python 3.9-3.12. Create a new environment with Python 3.11.

### Issue: Clients can't connect to server

**Solution**:

1. Make sure server is running first
2. Check firewall settings
3. Use `127.0.0.1:8080` for local testing

### Issue: "No BENIGN traffic found"

**Solution**: The selected datasets have been analyzed and contain benign traffic. Use the recommended files.

## 📊 Performance Metrics Explained

- **Accuracy**: Overall correct predictions
- **Precision**: Of all attacks detected, how many were real attacks?
- **Recall**: Of all real attacks, how many did we detect?
- **F1-Score**: Harmonic mean of precision and recall

For DDoS detection:

- **High Recall** is critical (must detect attacks)
- **High Precision** prevents false alarms

## 🎓 Project Submission Checklist

- [ ] Python 3.11 environment created
- [ ] All packages installed
- [ ] Model trains successfully
- [ ] Federated learning runs (server + 3 clients)
- [ ] Results documented (screenshots/logs)
- [ ] Literature survey included
- [ ] UI mockup/implementation
- [ ] Project report with architecture diagrams

## 📞 Quick Commands Reference

```powershell
# Analyze datasets
python analyze_datasets.py

# Test model
python model.py

# Run federated learning (4 terminals)
python server.py --rounds 5 --min-clients 3    # Terminal 1
python client.py 0                              # Terminal 2
python client.py 1                              # Terminal 3
python client.py 2                              # Terminal 4
```

## 🎯 Next Steps

1. **Fix Python version** (Use Python 3.11)
2. **Install TensorFlow and Flower**
3. **Run the system** (server + 3 clients)
4. **Document results** for your report
5. **Add UI dashboard** (optional but impressive)

---

**Made for**: VNR VJIET - Minor Project  
**Project**: Federated Learning for Decentralized DDoS Attack Detection in IoT Networks  
**Dataset**: CIC-DDoS 2019
