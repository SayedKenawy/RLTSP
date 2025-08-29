# Mathematical Model Documentation: RLTSP

## 1. State Representation and Management

### 1.1 State Representation
At each time step $t$, the state $s_t$ is defined as a sliding window of historical observations:

$$s_t = \begin{bmatrix} 
x_t \\
x_{t+1} \\
\vdots \\
x_{t+w-1}
\end{bmatrix} \in \mathbb{R}^{w \times 1}$$

Where:
- $w$ is the window size
- $x_t$ is the observation at time $t$
- The state is reshaped to $\mathbb{R}^{w \times 1}$ for processing

### 1.2 State Transition

$$s_{t+1} = \begin{bmatrix} 
x_{t+1} \\
x_{t+2} \\
\vdots \\
x_{t+w}
\end{bmatrix} = \text{shift}(s_t, -1) \circ [x_{t+1}, x_{t+2}, \dots, x_{t+w}]^T$$

## 2. Temporal Graph Convolutional Network (T-GCN)

### 2.1 Graph Convolution Operation
For each node $i$ at time $t$:

$$H_{i}^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_id_j}} H_j^{(l)} W^{(l)}\right)$$

### 2.2 Temporal GRU Update

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(Update Gate)}$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(Reset Gate)}$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad \text{(Candidate Activation)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(Hidden State Update)}$$

## 3. Multi-step Forecasting

### 3.1 Autoregressive Forecasting
Given input state $s_t$, the $k$-step ahead forecast $\hat{y}_{t+k|t}$ is computed as:

1. **Initial Step**:
   $$\hat{y}_{t+1|t} = f_\theta(s_t)$$
   
2. **Recursive Steps** for $k = 2$ to $K$:
   $$\hat{y}_{t+k|t} = f_\theta(\text{concat}(s_t[1:], \hat{y}_{t+1|t}, \dots, \hat{y}_{t+k-1|t}))$$

Where $f_\theta$ represents the T-GCN + forecast head model.

### 3.2 Forecast Head
$$\hat{y}_{t+1} = W_{\text{out}} \cdot \text{ReLU}(W_h \cdot h_t + b_h) + b_{\text{out}}}$$

## 4. Dueling Deep Q-Network (Dueling DQN)

### 4.1 Q-Value Decomposition
$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \left(A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a';\theta,\alpha)\right)$$

### 4.2 State-Action Value Update
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[r_{t+1} + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)\right]$$

## 5. Training Process

### 5.1 Loss Function (Huber Loss)
$$L(\theta) = \begin{cases} 
\frac{1}{2}(y - Q(s,a;\theta))^2 & \text{for } |y - Q(s,a;\theta)| \leq \delta \\
\delta(|y - Q(s,a;\theta)| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

### 5.2 Prioritized Experience Replay
- **TD-Error**: $\delta_i = r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)$
- **Priority**: $p_i = |\delta_i| + \epsilon$
- **Sampling Probability**: $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
- **Importance Sampling**: $w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$

## 6. Model Architecture Specifications

### 6.1 T-GCN Layer
- **Input**: $(N, T_{in}, F_{in}) = (1, w, 1)$
- **Hidden**: $(N, T_{out}, F_{out}) = (1, w, 16)$
- **Parameters**: $O(w \times 1 \times 32 + w \times 32 \times 16) = O(48w)$

### 6.2 Dueling Network
- **Value Stream**: $\mathbb{R}^{w} \xrightarrow{W_{128}} \mathbb{R}^{128} \xrightarrow{W_{64}} \mathbb{R}^{64} \xrightarrow{W_1} \mathbb{R}^{1}$
- **Advantage Stream**: $\mathbb{R}^{w} \xrightarrow{W'_{128}} \mathbb{R}^{128} \xrightarrow{W'_{64}} \mathbb{R}^{64} \xrightarrow{W'_3} \mathbb{R}^{|\mathcal{A}|}$

## 7. Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window Size ($w$) | 10 | Historical observations in state |
| Forecast Horizon | 5 | Number of steps to predict |
| $\gamma$ | 0.99 | Discount factor |
| $\epsilon_{\text{start}}$ | 1.0 | Initial exploration rate |
| $\epsilon_{\text{min}}$ | 0.01 | Minimum exploration rate |
| $\epsilon_{\text{decay}}$ | 0.995 | Exploration decay rate |
| $\alpha$ | 0.6 | Priority exponent |
| $\beta$ | 0.4 | Importance sampling exponent |
| $\tau$ | 0.01 | Target network update rate |
| Batch Size | 64 | Training batch size |
| Replay Buffer | 10,000 | Experience replay capacity |
| Learning Rate | 0.001 | Initial learning rate |
| Hidden Size | 32 | T-GCN hidden dimension |
| Output Size | 16 | T-GCN output dimension |
| Dropout | 0.2 | Dropout probability |

## 8. References
1. Wang et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", 2019
2. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", 2016
3. Schaul et al., "Prioritized Experience Replay", 2016
4. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014
5. [Graph-enabled Reinforcement Learning for Time Series Forecasting with Adaptive Intelligence](https://arxiv.org/pdf/2309.10186), 2023
