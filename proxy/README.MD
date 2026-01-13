# AI-Enhanced Multithreaded Proxy Server (C++)

## Overview

This is a **high-performance, multithreaded HTTP proxy server written in C++20** with **AI-driven classification and policy enforcement**. The proxy intercepts client connections, parses HTTP traffic, extracts behavioral features, consults an external AI service via UNIX domain sockets, applies policy decisions, and maintains structured logs for analysis.

**Key characteristics:**

* ⚡ Concurrent handling via fixed thread pool
* 🧠 Modular AI client for external classification services
* 🧱 Clean separation: networking, feature extraction, policy, logging
* 📊 CSV logging for post-analysis and model retraining
* 🐧 Linux-native using epoll for scalable I/O multiplexing

---

## Architecture

The High Level Design is as follows :

![AI-Driven Proxy Architecture](../assets/AI%20Driven%20Network%20Proxy%20HLD.png)

The proxy follows a layered architecture with clear separation of concerns:

```plaintext
Client Connection
    ↓
Proxy_Server (epoll multiplexing)
    ↓
Thread_Pool (distribute to workers)
    ↓
Feature_Extractor (parse HTTP → Feature_Vector)
    ↓
AI_Client (query external AI service)
    ↓
Policy_Engine (combine AI output + rules)
    ↓
CSV_Logger (record decision)
    ↓
Response/Drop Decision
```

---

## Components

### Proxy_Server

* **Purpose:** Core networking layer
* **Files:** [include/net/proxy_server.h](include/net/proxy_server.h), [src/net/proxy_server.cpp](src/net/proxy_server.cpp)
* **Details:** Listens on port 4005, uses epoll for non-blocking I/O, accepts connections, dispatches to thread pool

### Thread_Pool

* **Purpose:** Concurrent task execution
* **Files:** [include/thread/thread_pool.h](include/thread/thread_pool.h), [src/thread/thread_pool.cpp](src/thread/thread_pool.cpp)
* **Details:** Fixed worker threads, thread-safe task queue, graceful shutdown

### Feature_Extractor

* **Purpose:** Convert HTTP requests to structured features
* **Files:** [include/ai/feature_extractor.h](include/ai/feature_extractor.h), [src/ai/feature_extractor.cpp](src/ai/feature_extractor.cpp)
* **Features extracted:** payload size, header size, request count, inter-arrival time

### AI_Client

* **Purpose:** Interface with external AI classification service
* **Files:** [include/ai/ai_client.h](include/ai/ai_client.h), [src/ai/ai_client.cpp](src/ai/ai_client.cpp)
* **Communication:** UNIX domain socket to independent AI service
* **Returns:** Traffic type (BENIGN, BOT, ATTACK) with confidence score

### Policy_Engine

* **Purpose:** Decision-making based on AI and rules
* **Files:** [include/ai/policy_engine.h](include/ai/policy_engine.h), [src/ai/policy_engine.cpp](src/ai/policy_engine.cpp)
* **Functions:** Combines AI classification with deterministic policies, outputs allow/deny decisions

### CSV_Logger

* **Purpose:** Structured logging for analysis
* **Files:** [include/util/csv_logger.h](include/util/csv_logger.h), [src/util/csv_logger.cpp](src/util/csv_logger.cpp)
* **Output:** Writes decisions and metadata to CSV for offline review and model retraining

### Network Utilities

* **Purpose:** Common socket operations
* **Files:** [include/util/net_utils.h](include/util/net_utils.h), [src/util/net_utils.cpp](src/util/net_utils.cpp)
* **Helpers:** Non-blocking configuration, socket creation, error handling

### Common Definitions

* **File:** [include/common.h](include/common.h)
* **Defines:** `Feature_Vector`, `AI_Result`, `TRAFFIC_TYPE` enum

## Build & Run

### Requirements

* Linux (epoll, UNIX domain sockets)
* C++20 or later
* CMake ≥ 3.15
* GCC or Clang

### Build

```bash
cd proxy/
cmake -B build && cmake --build build
```

The executable `build/AI_Proxy` is created.

### Run

```bash
./build/AI_Proxy
```

Default: listens on `127.0.0.1:4005` with 8 worker threads. Adjust `PROXY_PORT` and thread count in [src/main.cpp](src/main.cpp).

### Testing the Proxy

**Terminal 1:** Start the proxy

```bash
./build/AI_Proxy
```

**Terminal 2:** Start the AI classifier (from `ai/scripts/`)

```bash
source ../.venv/bin/activate
python3 classifier.py
```

**Terminal 3:** Optional—monitor with mitmproxy

```bash
mitmproxy -p 8080
```

**Terminal 4:** Send test traffic

```bash
curl -v -x http://127.0.0.1:4005 http://example.com
```

## Extending the Proxy

### Add Custom Features

1. Extend `Feature_Vector` in [include/common.h](include/common.h)
2. Update extraction logic in [src/ai/feature_extractor.cpp](src/ai/feature_extractor.cpp)
3. Update AI service to accept the new fields

### Integrate Custom AI

The `AI_Client` communicates with an external service via UNIX socket. Swap the service by:

1. Modifying [src/ai/ai_client.cpp](src/ai/ai_client.cpp) serialization/deserialization
2. Updating the AI service listener socket path
3. Ensuring the AI service returns `AI_Result` (type + confidence)

### Modify Policies

Edit [src/ai/policy_engine.cpp](src/ai/policy_engine.cpp) to:

* Adjust classification thresholds
* Add traffic-type-specific rules
* Implement rate limiting or IP blocking

## Directory Layout

```bash
proxy/
├── CMakeLists.txt              # Build configuration
├── include/
│   ├── common.h                # Shared data structures
│   ├── ai/
│   │   ├── ai_client.h         # AI service interface
│   │   ├── feature_extractor.h # HTTP → Feature_Vector
│   │   └── policy_engine.h     # Classification + rules
│   ├── net/
│   │   └── proxy_server.h      # Main server (epoll)
│   ├── thread/
│   │   └── thread_pool.h       # Worker threads
│   └── util/
│       ├── csv_logger.h        # CSV output
│       └── net_utils.h         # Socket helpers
├── src/
│   ├── main.cpp                # Entry point
│   ├── ai/
│   │   ├── ai_client.cpp
│   │   ├── feature_extractor.cpp
│   │   └── policy_engine.cpp
│   ├── net/
│   │   └── proxy_server.cpp
│   ├── thread/
│   │   └── thread_pool.cpp
│   └── util/
│       ├── csv_logger.cpp
│       └── net_utils.cpp
├── build/                     # Build artifacts (generated)
│   └── AI_Proxy               # Compiled executable
├── proxy_metrics.csv          # Sample log output
└── README.md                  # This file
```

## License

This project is provided for educational and experimental purposes.

---

## Resources

* **Parent Directory:** See `../` for the full AI-driven proxy repository including Python classifier service
* **AI Service:** Refer to `../ai/scripts/classifier.py` for the external ML service
* **Datasets:** Check `../ai/data/` for training and evaluation data
