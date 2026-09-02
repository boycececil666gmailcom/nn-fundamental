# NN-Fundamental

> Modular from-scratch neural network implementation in Python and NumPy engineered to demonstrate core mathematical foundations, multi-layer forward propagation, and categorical classification dynamics.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

NN-Fundamental is a transparent, zero-dependency machine learning foundation library designed to illuminate neural network operations from raw matrix transformations to normalized probability distributions. It provides a crystal-clear reference architecture for developers and machine learning practitioners seeking deep, bottom-up mastery of artificial intelligence primitives.

---

## 1. End-to-End Pipeline Overview

```mermaid
---
config:
  theme: neutral
---
flowchart LR
    A["Raw Input Data"] --> B["Linear Transformation"]
    B --> C["Nonlinear Activation"]
    C --> D["Output Projection"]
    D --> E["Probability Normalization"]
    E --> F["Categorical Prediction"]
```

---

## 2. Core Purpose & Business Value

NN-Fundamental translates abstract mathematical modeling into high-trust operational assets by ensuring predictable computation, reducing third-party software risks, and guaranteeing full algorithmic transparency.

- **Predictable Operational Decision Making**: Delivers consistent, deterministic categorization of customer inputs and operational telemetry, empowering decision-makers with confident automated triage.
- **Zero-Dependency Cost Reduction**: Eliminates reliance on heavy external platform runtimes, significantly lowering hardware provisioning costs and eradicating cloud resource waste.
- **Flawless Enterprise Auditability**: Provides complete operational transparency where every intermediate data state can be mathematically inspected, satisfying governance and compliance benchmarks.
- **High-Velocity Prototyping & Retention**: Empowers engineering teams to rapidly test custom classification logic and validate assumptions without complex runtime setup delays.

---

## 3. System Architecture & Technical Execution

The architecture separates linear layer transformations, nonlinear activation gating, and probability normalization into decoupled modules to ensure numerical stability and rigorous state inspection.

### Core Concept & Phased Execution Sequence

```mermaid
---
config:
  theme: neutral
---
sequenceDiagram
    autonumber
    actor Caller as Inference Client
    participant FFN as Feedforward Network
    participant Act as Activation Module
    participant Norm as Normalization Engine

    Note over Caller, FFN: Phase 1: Input Ingestion & Hidden Transformation
    Caller->>FFN: forward(X) with batch input (N, input_dim)
    FFN->>FFN: Compute Affine Step: Z1 = X * W1 + b1
    FFN->>Act: Apply relu(Z1)
    Act-->>FFN: Return hidden activations H1 = max(0, Z1)

    Note over FFN, Norm: Phase 2: Output Projection & Softmax Normalization
    FFN->>FFN: Compute Projection Step: Z2 = H1 * W2 + b2
    FFN->>Norm: Apply softmax(Z2)
    alt Numerical Stability Check
        rect rgb(240, 243, 246)
            Norm->>Norm: Shift Logits: Z2 - max(Z2)
            Norm->>Norm: Compute Exp and Normalize Sum = 1.0
            Norm-->>FFN: Return Probability Distribution P
        end
    else Overflow Protection Fallback
        rect rgb(250, 235, 235)
            Norm->>Norm: Clamped Exponentiation
            Norm-->>FFN: Return Safe Probability Distribution P
        end
    end

    Note over FFN, Caller: Phase 3: Prediction & Class Assignment
    FFN->>FFN: Extract Argmax Class Indices
    FFN-->>Caller: Return Predictions & Class Probabilities
```

### High-Level Target Production Architecture

```mermaid
---
config:
  layout: elk
  theme: neutral
---
flowchart TB

    subgraph Ingestion["Input Ingestion Layer"]
        Batch["Feature Matrix X<br/>Shape: (Batch Size, Input Dimension)"]
    end

    subgraph HiddenLayer["Hidden Representation Layer"]
        Weight1["Weight Matrix W1<br/>Shape: (Input, Hidden)"]
        Bias1["Bias Vector b1<br/>Shape: (1, Hidden)"]
        Affine1["Linear Map: X * W1 + b1"]
        Relu["Activation Function: ReLU"]
    end

    subgraph OutputLayer["Output Projection Layer"]
        Weight2["Weight Matrix W2<br/>Shape: (Hidden, Output)"]
        Bias2["Bias Vector b2<br/>Shape: (1, Output)"]
        Affine2["Linear Map: H1 * W2 + b2"]
        Softmax["Normalization: Softmax"]
    end

    subgraph Evaluation["Prediction & Verification Layer"]
        Probabilities["Probability Distribution<br/>Row Sum = 1.0"]
        Argmax["Argmax Class Assignment"]
    end

    Batch --> Affine1
    Weight1 --> Affine1
    Bias1 --> Affine1
    Affine1 --> Relu
    Relu --> Affine2
    Weight2 --> Affine2
    Bias2 --> Affine2
    Affine2 --> Softmax
    Softmax --> Probabilities
    Probabilities --> Argmax
```

### Execution Boundary & Component Isolation Design

```mermaid
---
config:
  layout: elk
  theme: neutral
---
flowchart TB

    subgraph External["Client Execution Domain"]
        Script["Application Runtime / Evaluation Script"]
    end

    subgraph Runtime["Network Execution Boundary"]

        subgraph MathOps["Mathematical Primitives (Activation Functions)"]
            SigmoidFn["sigmoid(x) = 1 / (1 + exp(-x))"]
            ReluFn["relu(x) = max(0, x)"]
            SoftmaxFn["softmax(x) = exp(x - max) / sum(exp(x - max))"]
        end

        subgraph ModelCore["SimpleFeedforwardNetwork"]
            Init["__init__(input_size, hidden_size, output_size)<br/>Parameters: W1, b1, W2, b2"]
            Forward["forward(X: np.ndarray) -> np.ndarray"]
        end

    end

    Script -->|"Input Tensor (5, 10)"| Forward
    Init -->|"Param Storage"| Forward
    Forward -->|"Compute Hidden"| ReluFn
    Forward -->|"Compute Output"| SoftmaxFn
    Forward -->|"Probability Matrix (5, 3)"| Script
```

---

## 4. Repository Structure

```text
NN-Fundamental/
├── 1.FFN/
│   └── main.py          # Feedforward neural network, activations, and test execution
├── 2.CNN/
│   └── main.py          # Convolutional neural network, spatial pooling, and test execution
├── 3.RNN/
│   └── main.py          # Recurrent neural network, sequence dynamics, and test execution
├── .gitignore           # Environment and cache artifact ignore rules
├── pyproject.toml       # Project metadata and dependency specification
├── requirements.txt     # Standard dependency pin (numpy>=1.24.0)
├── ruff.toml            # Linter and formatter configuration
├── uv.lock              # Deterministic uv dependency lockfile
└── README.md            # Architecture documentation and execution guide
```

---

## 5. Quickstart

### Setup & Synchronization with `uv`

```powershell
# Sync virtual environment and dependencies
uv sync
```

### Run Demonstrations

```powershell
# Run Feedforward Network
uv run python 1.FFN/main.py

# Run Convolutional Neural Network
uv run python 2.CNN/main.py

# Run Recurrent Neural Network
uv run python 3.RNN/main.py
```

