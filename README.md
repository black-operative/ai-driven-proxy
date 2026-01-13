# AI-Driven Network Proxy

A comprehensive system combining a high-performance C++ proxy server with an AI classification service to detect and manage malicious network traffic in real-time.

## 📋 Project Overview

This project implements an **AI-enhanced multithreaded proxy server** that:

- **Intercepts** HTTP traffic on port 4005
- **Extracts** network features from each request
- **Classifies** traffic as BENIGN, BOT, or ATTACK using a trained ML model
- **Enforces** policies based on classification confidence
- **Logs** decisions for analysis and model retraining

The system is split into two main components:

1. **Proxy Server** (C++20) — High-performance networking with epoll multiplexing
2. **AI Classifier** (Python) — Real-time ML inference via socket communication

---

## 🏗️ Architecture

The High Level Design is as follows :

![AI-Driven Proxy Architecture](assets/AI%20Driven%20Network%20Proxy%20HLD.png)

```plaintext
Client Connection
    ↓
Proxy_Server            (epoll multiplexing, port 4005)
    ↓
Thread_Pool             (8 worker threads by default)
    ↓
Feature_Extractor       (HTTP → Feature_Vector)
    ↓
AI_Client               (TCP socket to classifier service, port 5000)
    ↓
Policy_Engine           (combine AI output + deterministic rules)
    ↓
CSV_Logger              (record decision to proxy_metrics.csv)
    ↓
Response/Drop Decision
```

**Key Design Principles:**

- Non-blocking I/O for scalability
- Thread-safe feature extraction and logging
- Modular AI interface (swap classifiers without recompiling proxy)
- Real-time latency tracking (feature extraction, AI inference, policy decision)

---

## 📁 Directory Structure

```bash
.
├── README.md                              # This file
├── assets/                                # Architecture diagrams
├── proxy/                                 # C++ proxy server
│   ├── CMakeLists.txt
│   ├── README.MD                          # Detailed proxy documentation
│   ├── include/
│   │   ├── common.h                       # Shared structs (Feature_Vector, AI_Result)
│   │   ├── ai/
│   │   │   ├── ai_client.h
│   │   │   ├── feature_extractor.h
│   │   │   └── policy_engine.h
│   │   ├── net/
│   │   │   └── proxy_server.h
│   │   ├── thread/
│   │   │   └── thread_pool.h
│   │   └── util/
│   │       ├── csv_logger.h
│   │       └── net_utils.h
│   ├── src/
│   │   ├── main.cpp
│   │   ├── ai/
│   │   │   ├── ai_client.cpp
│   │   │   ├── feature_extractor.cpp
│   │   │   └── policy_engine.cpp
│   │   ├── net/
│   │   │   └── proxy_server.cpp
│   │   ├── thread/
│   │   │   └── thread_pool.cpp
│   │   └── util/
│   │       ├── csv_logger.cpp
│   │       └── net_utils.cpp
│   ├── build/                             # CMake build artifacts
│   │   └── AI_Proxy                       # Compiled executable
│   └── proxy_metrics.csv                  # Sample metrics log
└── ai/                                    # Python ML pipeline
    ├── README.md                          # AI service documentation
    ├── scripts/
    │   ├── classifier.py                  # Real-time inference service
    │   ├── cicids_to_dataset.py           # Dataset preprocessing
    │   └── analysis.py                    # Performance visualization
    ├── data/
    │   ├── dataset.csv                    # Normalized training data
    │   ├── Friday-DDOS.csv                # CICIDS raw data (DDoS)
    │   ├── Tuesday-Normal.csv             # CICIDS raw data (benign)
    │   └── copy.csv                       # Working copy
    └── model/
        └── model.pkl                      # Trained scikit-learn classifier
```

---

## 🚀 Quick Start

### Prerequisites

**Proxy:**

- Linux (epoll, UNIX domain sockets required)
- C++20 compiler (GCC 10+ or Clang 10+)
- CMake ≥ 3.15

**AI Classifier:**

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib

### Build & Run

#### 1. Build the Proxy

```bash
cd proxy/
cmake -B build && cmake --build build
```

Output: AI_Proxy executable

#### 2. Install AI Dependencies

```bash
cd ai/
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib
```

#### 3. Start the Classifier Service

```bash
cd ai/scripts/
python3 classifier.py
```

Expected output:

```bash
Listening on port 5000...
```

#### 4. Start the Proxy (in another terminal)

```bash
cd proxy/
./build/AI_Proxy
```

Expected output:

```bash
Proxy listening on 127.0.0.1:4005...
```

#### 5. Test with Traffic

```bash
cd proxy/
curl -v -x http://127.0.0.1:4005 http://example.com
```

Check results in proxy_metrics.csv.

---

## 📊 Data Formats

### Feature Vector (Proxy → AI)

Extracted from each HTTP request:

| Field | Type | Unit | Range |
|-------|------|------|-------|
| `payload_size` | uint32 | bytes | 0 – 4294967295 |
| `header_size` | uint32 | bytes | 0 – 4294967295 |
| `request_count` | uint32 | count | 0 – 4294967295 |
| `inter_arrival_us` | uint64 | microseconds | 0 – 18446744073709551615 |

**Binary format:** 24 bytes (IIIQ in struct layout)

### AI Result (AI → Proxy)

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `type` | uint8 | 0–2 | 0=BENIGN, 1=BOT, 2=ATTACK |
| `confidence` | float32 | 0.0–1.0 | Model confidence score |

**Binary format:** 5 bytes

### CSV Log Schema

```csv
timestamp_us,client_fd,feature_us,ai_us,policy_us,total_us,decision,traffic_class,confidence
1696540800000000,15,45,1200,30,1275,1,0,0.987
```

Fields:

- `timestamp_us` — Request timestamp (microseconds)
- `client_fd` — File descriptor of client socket
- `feature_us` — Time to extract features
- `ai_us` — Time for AI inference
- `policy_us` — Time for policy decision
- `total_us` — Total processing time
- `decision` — 0=allow, 1=drop
- `traffic_class` — 0=BENIGN, 1=BOT, 2=ATTACK
- `confidence` — AI model confidence (0.0–1.0)

---

## 🛠️ Configuration

### Proxy Configuration

Edit main.cpp:

```cpp
const int PROXY_PORT = 4005;        // Listen port
const int NUM_WORKERS = 8;          // Thread pool size
const int BACKLOG = 128;            // Listen backlog
```

### AI Classifier Configuration

Edit classifier.py:

```python
PORT = 5000                         # Listen port
MODEL_PATH = "../model/model.pkl"   # Model file
```

### Policy Thresholds

Edit policy_engine.cpp:

```cpp
const float CONFIDENCE_THRESHOLD = 0.7f;  // Drop if confidence > threshold
const float BENIGN_THRESHOLD = 0.5f;      // Allow benign traffic
```

---

## 📈 Performance Characteristics

**Typical latencies** (measured on Linux x86-64):

| Component | Time | Notes |
|-----------|------|-------|
| Feature extraction | 40–80 µs | HTTP parsing |
| AI inference | 500–2000 µs | Model depends on classifier |
| Policy decision | 10–50 µs | Rule evaluation |
| **Total** | **600–2100 µs** | Per request |
| Socket I/O overhead | 50–70 µs | TCP communication |

See analysis.py for detailed histograms.

---

## 🔌 Integration Details

### Proxy → AI Communication

1. Proxy extracts `Feature_Vector` from HTTP request
2. Proxy serializes to 24 bytes via struct packing
3. Proxy sends via TCP socket to `localhost:5000`
4. Classifier receives, deserializes, predicts
5. Classifier sends 5 bytes back (type + confidence)
6. Proxy deserializes `AI_Result`
7. Policy_Engine makes decision

### Model Requirements

The classifier expects a scikit-learn estimator with:

- `.predict(X)` → returns class labels (0, 1, or 2)
- `.predict_proba(X)` → returns confidence scores

Example training:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

X = ... # feature matrix (4 columns)
y = ... # labels (0, 1, 2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
```

---

## 🔧 Extending the System

### Add Custom Features

1. Extend `Feature_Vector` in common.h
2. Update extraction in feature_extractor.cpp
3. Update AI classifier input shape
4. Rebuild: `cmake --build proxy/build`

### Swap the AI Classifier

1. Modify socket protocol in ai_client.cpp
2. Implement new serialization/deserialization
3. Update classifier.py to accept new format
4. Test with curl (step 5 above)

### Modify Policies

Edit policy_engine.cpp:

- Adjust classification thresholds
- Add traffic-type-specific rules
- Implement rate limiting or IP blocking
- Rebuild and redeploy

### Retrain the Model

Use cicids_to_dataset.py to preprocess raw CICIDS data:

```bash
cd ai/scripts/
python3 cicids_to_dataset.py
# Outputs: dataset.csv
```

Then train a new model and save to `model.pkl`.

---

## 📊 Analyzing Results

### Generate Performance Report

```bash
cd ai/scripts/
python3 analysis.py
```

Outputs:

- Confidence trajectory chart
- Latency box plots
- Confidence-latency correlation scatter
- Decision distribution pie chart

### Manual CSV Inspection

```bash
head -20 proxy/proxy_metrics.csv
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Proxy build fails | Ensure C++20 support: `g++ --version` |
| "Connection refused" | Check AI classifier is running on port 5000 |
| "Serialization error" | Verify struct sizes match (24 bytes input, 5 output) |
| "No model.pkl" | Train model or provide pre-trained weights |
| High latencies | Check system load, AI model complexity |
| CSV not created | Ensure proxy directory is writable |

---

## 📚 Documentation

- **README.MD** — Detailed proxy architecture, components, and usage
- **README.md** — AI pipeline, model training, classifier service details
- **assets** — Architecture diagrams and design documents

---

## 📋 Dataset Information

The system uses the **CICIDS2017** dataset:

- **Friday-DDOS.csv** — DDoS attack traffic
- **Tuesday-Normal.csv** — Normal benign traffic
- **dataset.csv** — Preprocessed and normalized for training

Features extracted:

- `payload_size` — Total forward packet length
- `header_size` — Forward packet header length
- `request_count` — Total forward packets
- `inter_arrival_us` — Inter-arrival time (ms → µs)

Classes:

- **BENIGN** — Legitimate traffic
- **BOT** — Bot-generated traffic
- **ATTACK** — Malicious attack traffic

---

## 🔬 Technical Specifications

### Proxy Server

- **Language:** C++20
- **I/O Model:** epoll (Linux)
- **Threading:** Thread pool with configurable workers
- **Concurrency:** Lock-free feature extraction, mutex-protected logging
- **Protocol:** HTTP/1.1 transparent proxy
- **Max Connections:** Depends on file descriptor limits

### AI Classifier

- **Language:** Python 3.7+
- **Framework:** scikit-learn
- **Model Format:** Pickle (joblib-compatible)
- **Communication:** TCP socket (localhost:5000)
- **Serialization:** struct packing (binary)

---

## 📝 Example Workflow

### Training a New Model

```bash
# 1. Preprocess CICIDS data
cd ai/scripts/
python3 cicids_to_dataset.py

# 2. Train classifier (implement in train.py or Jupyter)
python3 train.py  # Creates model.pkl

# 3. Restart classifier service
python3 classifier.py

# 4. Restart proxy
cd ../../proxy/
./build/AI_Proxy

# 5. Send test traffic
curl -x http://127.0.0.1:4005 http://example.com

# 6. Analyze results
cd ../ai/scripts/
python3 analysis.py
```

---

## ⚖️ License

This project is provided for educational and experimental purposes.

---

## 📞 Support & Contributions

- **Issues:** Check README.MD and README.md for component-specific help
- **Troubleshooting:** See section above
- **Architecture questions:** Refer to High-Level Design diagram in assets

---

## 🔗 Related Resources

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Linux epoll Manual](https://man7.org/linux/man-pages/man7/epoll.7.html)
- [C++20 Features](https://en.cppreference.com/w/cpp/compiler_support/20)

---

**Last Updated:** 2024  
**Project:** AI-Driven Network Proxy  
**Status:** Active Development
