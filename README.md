# Reinforcement Learning for Time Series Prediction (RLTSP)

This project implements an advanced Reinforcement Learning framework for Time Series Prediction (RLTSP) using PyTorch. The framework combines Temporal Graph Convolutional Networks (T-GCN) with Dueling Deep Q-Learning to predict and trade on time series data with adaptive intelligence.

## ✨ Key Features

- **Temporal Graph Convolutional Networks (T-GCN)** for capturing spatial and temporal dependencies in time series data
- **Dueling Deep Q-Network (DDQN)** for optimal policy learning with prioritized experience replay
- **Advanced State Representation** with sliding window approach for time series modeling
- **Mathematically Rigorous** implementation with detailed documentation
- **Modular Architecture** for easy extension and customization
- **Comprehensive Data Pipeline** with built-in preprocessing and normalization

## 🏗️ Project Structure

```
RLTSP/
├── RLTSP.py                 # Main implementation of the RLTSP framework
├── data_loader.py           # Data loading and preprocessing utilities
├── requirements.txt         # Python dependencies
├── MATHEMATICAL_MODEL.md    # Detailed mathematical documentation
└── README.md                # This file
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RLTSP
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Core Concepts

### State Representation
- Sliding window of historical observations
- Normalized state space for stable training
- Adaptive state transition dynamics

### Model Architecture
- **T-GCN Layer**: Captures spatial and temporal patterns
- **GRU Network**: Models sequential dependencies
- **Dueling DQN**: Separates value and advantage streams

## 🛠️ Usage

### Data Preparation
1. Prepare your time series data as a NumPy array
2. Configure data loading in `data_loader.py`
3. Set appropriate hyperparameters in the configuration

### Training
```python
from RLTSP import OptimizedRLTSPFramework
import numpy as np

# Load your data
data = np.load('your_data.npy')

# Initialize the framework
model = OptimizedRLTSPFramework(
    data=data,
    window_size=10,
    forecast_horizon=5
)

# Train the model
model.train(episodes=200)
```

### Prediction
```python
# Make predictions
predictions = model.predict(input_data, steps=5)

# Evaluate performance
performance = model.evaluate(test_data)
```

## 📊 Performance

- **Training Stability**: Advanced techniques like gradient clipping and learning rate scheduling
- **Sample Efficiency**: Prioritized experience replay for better data utilization
- **Convergence**: Fast convergence with dueling architecture and target networks

## 📚 Documentation

For detailed mathematical formulation and implementation details, please refer to:
- [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md)
- Inline code documentation

## 📝 Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy
- Pandas
- Matplotlib (for visualization)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Wang et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", 2019
2. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", 2016
3. [Graph-enabled Reinforcement Learning for Time Series Forecasting](https://arxiv.org/pdf/2309.10186), 2023
