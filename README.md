# FedMD-XAI
# FedMD-XAI: Privacy-Preserving & Explainable Mobile Malware Detection

A comprehensive framework implementing **Federated Learning** with **Differential Privacy** and **Explainable AI (XAI)** for mobile malware detection. This system combines multiple machine learning approaches to achieve high accuracy while preserving data privacy and providing model interpretability.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Models & Algorithms](#models--algorithms)
- [Privacy Features](#privacy-features)
- [Explainability](#explainability)
- [Performance](#performance)
- [Performance Monitoring](#performance-monitoring)
- [Contributing](#contributing)

## Features
### Core Capabilities
- **Federated Learning**: Privacy-preserving collaborative training across multiple clients
- **Differential Privacy**: Mathematical privacy guarantees with configurable privacy budget (epsilon)
- **Explainable AI**: SHAP and LIME explanations for model predictions
- **Multimodal Detection**: Static, dynamic, and bytecode feature analysis
- **Ensemble Models**: Optimized combination of Random Forest, SVM, DNN, and Logistic Regression

### Key Innovations
- **FedMD-XAI**: Novel framework combining federated learning with model distillation and explainability
- **Z-Score Anomaly Detection**: Automatic detection of malicious client updates
- **Class Imbalance Handling**: Balanced class weights and synthetic data augmentation
- **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation

## Architecture

```
FedMD-XAI Framework
    |
    |-- Federated Learning Server
    |   |-- Client Selection (10/50 clients per round)
    |   |-- Model Aggregation (FedAvg)
    |   |-- Anomaly Detection (Z-score threshold)
    |   `-- Privacy Protection (Differential Privacy)
    |
    |-- Ensemble Malware Detector
    |   |-- Random Forest (200 trees, max_depth=25)
    |   |-- SVM (RBF kernel, C=10.0)
    |   |-- Deep Neural Network (256-128-64-32)
    |   `-- Logistic Regression (C=10.0)
    |
    |-- XAI Explainer
    |   |-- SHAP explanations (global & local)
    |   |-- LIME explanations (local)
    |   `-- Feature importance analysis
    |
    `-- Multimodal Feature Extractor
        |-- Static Features (152 dimensions)
        |-- Dynamic Features (15 dimensions)
        `-- Bytecode Features (256 dimensions)
```

## Installation
### Prerequisites
- Python 3.8+
- Streamlit
- Required ML libraries

### Setup
1. **Clone repository**
```bash
git clone https://github.com/TakudzwaChoto/FedMD-XAI.git
cd fedmd-xai
```

2. **Install dependencies**
```bash
# Use the compatible requirements file
pip install -r requirements_compatible.txt

# Or install manually
pip install streamlit==1.32.0 pandas==2.2.2 numpy==1.26.4
pip install matplotlib==3.8.4 seaborn==0.13.2 scikit-learn==1.4.2
pip install shap==0.46.0 lime==0.2.0.1 psutil==5.9.8
```

3. **Run the application**
```bash
streamlit run main.py --server.port 8501
```

### Requirements Files
- `requirements_compatible.txt` - Recommended version with resolved dependencies
- `requirements_new.txt` - Basic requirements list
- `requirements_fixed.txt` - Alternative configuration

## Usage
### 1. Data Upload
- Supported formats: CSV, TSV, pipe-delimited (|)
- Automatic label column detection
- Support for various label names: `legitimate`, `Label`, `class`, `malware`, etc.

### 2. Model Configuration
- **Federated Learning Settings**:
  - Number of rounds: 10
  - Number of clients: 50
  - Clients per round: 10
  - Local epochs: 3
  
- **Privacy Settings**:
  - Epsilon (privacy budget): 0.5
  - Delta: 1e-5
  - Gradient clipping: 1.5

### 3. Training Process
1. **Data Preprocessing**: Automatic feature extraction and normalization
2. **Federated Training**: Distributed learning across simulated clients
3. **Model Evaluation**: Comprehensive metrics and cross-validation
4. **XAI Analysis**: Generate explanations for predictions

### 4. Results Analysis
- Performance metrics (Accuracy, Precision, Recall, F1, AUC)
- Confusion matrix and classification report
- Feature importance rankings
- SHAP and LIME visualizations

### 5. Performance Monitoring
- **Training Time**: Per-round and total training time measurement
- **Network Latency**: Upload/download/round-trip communication time
- **Communication Cost**: Model update size tracking (MB per round)
- **Device Performance**: Memory and CPU usage monitoring
- **System Stability**: Client dropout simulation and resilience testing
- **XAI Latency**: Explanation generation time measurement
- **DP Overhead**: Differential privacy performance impact analysis

## Configuration
### Federated Learning Config
```python
@dataclass
class FederatedLearningConfig:
    num_rounds: int = 10
    num_clients: int = 50
    clients_per_round: int = 10
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    z_score_threshold: float = 2.5
```

### Differential Privacy Config
```python
@dataclass
class DifferentialPrivacyConfig:
    epsilon: float = 0.5
    delta: float = 1e-5
    gradient_clip_norm: float = 1.5
    noise_scale: float = 1.0
```

### Ensemble Model Config
```python
@dataclass
class EnsembleModelConfig:
    random_forest: Dict = {
        'n_estimators': 200,
        'max_depth': 25,
        'class_weight': 'balanced'
    }
    svm: Dict = {
        'kernel': 'rbf',
        'C': 10.0,
        'class_weight': 'balanced'
    }
    neural_network: Dict = {
        'hidden_layer_sizes': (256, 128, 64, 32),
        'max_iter': 500
    }
    ensemble_weights: Dict = {
        'rf': 0.30, 'svm': 0.25, 'dnn': 0.25, 'lr': 0.20
    }
```

## Models & Algorithms
### 1. Random Forest
- **Parameters**: 200 trees, max depth 25
- **Features**: Gini impurity, balanced class weights
- **Purpose**: Feature importance extraction, robust baseline

### 2. Support Vector Machine
- **Kernel**: RBF (Radial Basis Function)
- **Regularization**: C=10.0
- **Purpose**: Non-linear pattern detection

### 3. Deep Neural Network
- **Architecture**: 256-128-64-32 layers
- **Activation**: ReLU
- **Optimizer**: Adam with early stopping
- **Purpose**: Complex pattern recognition

### 4. Logistic Regression
- **Regularization**: L2 with C=10.0
- **Solver**: LBFGS
- **Purpose**: Linear decision boundary, interpretability

## Privacy Features
### Differential Privacy
- **Mechanism**: Gaussian noise addition to gradients
- **Privacy Budget**: Epsilon=0.5 (configurable)
- **Gradient Clipping**: Norm limit of 1.5
- **Composition**: Privacy loss tracked across rounds

### Federated Learning Security
- **Client Selection**: Random subset per round
- **Anomaly Detection**: Z-score threshold for malicious updates
- **Secure Aggregation**: FedAvg algorithm with privacy protection

## Explainability
### SHAP (SHapley Additive exPlanations)
- **Global Explanations**: Overall feature importance
- **Local Explanations**: Individual prediction explanations
- **Visualization**: Force plots and summary charts

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local Fidelity**: Instance-specific explanations
- **Feature Perturbation**: Local surrogate models
- **Interpretability**: Linear explanations for complex models

### Feature Analysis
- **Top Features**: Ranked by importance scores
- **Decision Boundaries**: Visualization of classification logic
- **Pattern Recognition**: Identification of malware indicators

## Performance
### Expected Metrics
#### Depending on devices and dataset
- **Accuracy**: >90%
- **Precision**: >92%
- **Recall**: >88%
- **F1-Score**: >90%
- **AUC-ROC**: >0.95

## Performance Monitoring
### MUST-MEASURE Metrics
The FedMD-XAI framework includes comprehensive performance monitoring capabilities to evaluate real-world feasibility:

1. **Training Time per FL Round**
   - Measures time taken for each federated learning round
   - Tracks total training time across all rounds
   - Identifies performance bottlenecks in training process

2. **Network Latency (Communication Time)**
   - Upload time: Client to server model update transmission
   - Download time: Server to client model distribution
   - Round-trip latency: Complete communication cycle measurement

3. **Model Update Size (Communication Cost)**
   - Measures size of model updates transmitted per round
   - Tracks bandwidth requirements for federated learning
   - Calculates total communication overhead

4. **Device Performance (Resource Usage)**
   - Memory usage: RAM consumption during training
   - CPU usage: Processor utilization percentage
   - Peak resource tracking for capacity planning

5. **System Stability (Client Dropout Test)**
   - Simulates client disconnections during training
   - Measures system resilience to client failures
   - Tracks recovery and adaptation capabilities

6. **XAI Latency (Explanation Time)**
   - Measures SHAP explanation generation time
   - Tracks LIME local explanation performance
   - Evaluates real-time explanation feasibility

### Optional Metrics
- **Differential Privacy Overhead**: Performance impact of privacy mechanisms
- **Final Accuracy**: Model performance validation

### Performance Dashboard
- Real-time visualization of all metrics
- 8-panel comprehensive dashboard
- Export capabilities for research documentation
- Performance recommendations based on measurements

### Logging Format
```
=== MUST-MEASURE PERFORMANCE METRICS ===
1. Training Time per FL Round
   Round times: [45.2s, 42.8s, 40.1s, ...]
   Average: 42.7s/round
   Total: 427.0s

2. Network Latency (Communication Time)
   Upload time: 245.3 ms
   Download time: 189.7 ms
   Round-trip: 435.0 ms
```

### Evaluation Methods
- **5-fold Cross-Validation**: Robust performance estimation
- **Stratified Sampling**: Preserves class distribution
- **Multiple Metrics**: Comprehensive evaluation suite
- **Statistical Tests**: Confidence intervals and significance

### Optimization Features
- **Class Balance**: Handles imbalanced datasets
- **Feature Selection**: Automatic feature importance ranking
- **Hyperparameter Tuning**: Optimized configurations
- **Early Stopping**: Prevents overfitting

## Advanced Features
### Model Distillation
- Knowledge transfer from ensemble to lightweight models
- Maintains performance while reducing complexity
- Supports deployment on resource-constrained devices

### Anomaly Detection
- Z-score based detection of malicious clients
- Automatic exclusion of outlier updates
- Protection against adversarial attacks

### Data Augmentation
- Synthetic sample generation for minority classes
- SMOTE-like techniques for imbalance handling
- Improved generalization performance

## File Structure

```
fedmd-xai/
    |
    |-- main.py                 # Main Streamlit application
    |-- README.md              # This documentation
    |
    |-- Core Components/
    |   |-- Federated Learning Server
    |   |-- Ensemble Models
    |   |-- XAI Explainers
    |   `-- Privacy Modules
    |
    |-- Configuration/
    |   |-- DifferentialPrivacyConfig
    |   |-- FederatedLearningConfig
    |   |-- MultimodalConfig
    |   `-- EnsembleModelConfig
    |
    `-- Utilities/
        |-- Data Preprocessing
        |-- Feature Extraction
        `-- Visualization Tools
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
```
**Note**: This framework is designed for research and educational purposes. For production deployment, ensure proper security measures and compliance with relevant regulations.
