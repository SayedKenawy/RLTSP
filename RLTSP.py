import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Tuple, List, Dict, Optional
import warnings

# Import data loading functions
from data_loader import load_data, load_test_data

warnings.filterwarnings('ignore')
import copy
import time
import math

# Define experience tuple for prioritized replay
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done', 'priority'])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        experience = Experience(state, action, reward, next_state, done, self.max_priority)
        self.buffer[self.position] = experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences with priority weights"""
        if len(self.buffer) == 0:
            return [], [], [], [], [], []

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones, priorities = zip(*experiences)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, np.array(weights))

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class OptimizedTemporalGraphConvLayer(nn.Module):
    """
    Optimized Temporal Graph Convolutional Layer for faster training
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_nodes: int, dropout=0.1):
        super(OptimizedTemporalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_nodes = num_nodes

        # Spatial GCN layers with residual connections
        self.gcn1 = nn.Linear(input_size, hidden_size)
        self.gcn2 = nn.Linear(hidden_size, output_size)
        self.residual = nn.Linear(input_size, output_size) if input_size != output_size else None

        # Temporal GRU layer
        self.gru = nn.GRU(output_size, output_size, batch_first=True, dropout=dropout)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through T-GCN layer
        """
        if adj_matrix is None:
            adj_matrix = torch.eye(self.num_nodes, device=x.device)

        # Ensure input has 4 dimensions [batch, seq, nodes, features]
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # Add feature dimension

        batch_size, seq_len, num_nodes, _ = x.shape

        # Spatial convolution
        x_reshaped = x.reshape(-1, num_nodes, self.input_size)

        # Apply GCN with residual connection
        spatial_out = F.relu(self.norm1(self.gcn1(x_reshaped)))
        spatial_out = self.dropout(spatial_out)
        spatial_out = self.gcn2(spatial_out)

        # Add residual connection if needed
        if self.residual is not None:
            residual = self.residual(x_reshaped)
            spatial_out = spatial_out + residual
        else:
            spatial_out = spatial_out + x_reshaped

        spatial_out = torch.matmul(adj_matrix, spatial_out)
        spatial_out = torch.tanh(spatial_out)

        # Reshape for temporal processing
        spatial_out = spatial_out.view(batch_size, seq_len, num_nodes, self.output_size)

        # Temporal processing with GRU
        temporal_out = []
        for i in range(num_nodes):
            node_data = spatial_out[:, :, i, :]  # [batch_size, seq_len, output_size]
            node_data = self.norm2(node_data)
            gru_out, _ = self.gru(node_data)
            temporal_out.append(gru_out.unsqueeze(2))

        output = torch.cat(temporal_out, dim=2)
        return output


class OptimizedPredictiveEnvironment:
    """
    Optimized RLTSP Predictive Environment for faster training
    """

    def __init__(self, data: np.ndarray, window_size: int = 10, forecast_horizon: int = 5):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.current_step = 0
        self.max_steps = len(data) - window_size - forecast_horizon

        # State and action spaces
        self.state_dim = 1  # Simplified for 1D data
        self.action_dim = 3  # Hold, Buy, Sell

        # Optimized T-GCN model for forecasting
        self.num_nodes = 1  # Simplified for 1D data
        self.tgcn = OptimizedTemporalGraphConvLayer(
            input_size=1,
            hidden_size=32,  # Increased hidden size
            output_size=16,  # Increased output size
            num_nodes=self.num_nodes,
            dropout=0.2
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )

        # Training components
        self.optimizer = optim.Adam(
            list(self.tgcn.parameters()) + list(self.forecast_head.parameters()),
            lr=0.001
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.loss_fn = nn.HuberLoss()  # More robust loss function

        # For tracking performance
        self.forecast_errors = []

    def get_state(self) -> np.ndarray:
        """Get current state (historical window)"""
        return self.data[self.current_step:self.current_step + self.window_size].reshape(-1, 1)

    def get_model_state(self):
        """Get the state of the forecasting model"""
        return {
            'tgcn_state': copy.deepcopy(self.tgcn.state_dict()),
            'forecast_head_state': copy.deepcopy(self.forecast_head.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler_state': copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else None
        }

    def set_model_state(self, state_dict):
        """Set the state of the forecasting model"""
        self.tgcn.load_state_dict(state_dict['tgcn_state'])
        self.forecast_head.load_state_dict(state_dict['forecast_head_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if self.scheduler and state_dict['scheduler_state']:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])

    def forecast_future_states(self, state: np.ndarray, num_steps: Optional[int] = None) -> np.ndarray:
        """
        Forecast future states using T-GCN with teacher forcing during training
        """
        if num_steps is None:
            num_steps = self.forecast_horizon

        self.tgcn.eval()
        self.forecast_head.eval()
        with torch.no_grad():
            # Prepare input tensor with correct dimensions [batch, seq, nodes, features]
            x = torch.FloatTensor(state).unsqueeze(0).unsqueeze(-1)  # [1, window_size, 1, 1]

            # Forward through T-GCN
            tgcn_out = self.tgcn(x)  # [batch_size, seq_len, num_nodes, hidden_size]

            # Use the last timestep output for forecasting
            last_output = tgcn_out[:, -1, :, :]  # [batch_size, num_nodes, hidden_size]

            # Generate multi-step forecasts using autoregressive approach
            forecasts = []
            current_hidden = last_output

            for _ in range(num_steps):
                # Generate forecast for next step
                forecast = self.forecast_head(current_hidden)  # [batch_size, num_nodes, 1]
                forecasts.append(forecast.squeeze().cpu().numpy())

                # Update hidden state for next prediction using a gating mechanism
                current_hidden = 0.8 * current_hidden + 0.2 * forecast

            return np.array(forecasts)

    def train_forecaster(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Train the T-GCN forecaster with teacher forcing
        """
        self.tgcn.train()
        self.forecast_head.train()
        self.optimizer.zero_grad()

        # Forward pass
        tgcn_out = self.tgcn(X)  # [batch_size, seq_len, num_nodes, hidden_size]
        last_output = tgcn_out[:, -1, :, :]  # [batch_size, num_nodes, hidden_size]
        forecast = self.forecast_head(last_output)  # [batch_size, num_nodes, 1]
        forecast = forecast.squeeze()  # [batch_size]

        # Ensure y has the right shape
        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Compute loss
        loss = self.loss_fn(forecast, y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.tgcn.parameters()) + list(self.forecast_head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        return loss.item()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return new state, reward, done, info
        """
        current_state = self.get_state()
        forecast_error = 0
        predicted_future = None
        current_price = current_state[-1, 0]

        # Get actual future values for reward calculation
        if self.current_step + self.window_size + self.forecast_horizon <= len(self.data):
            actual_future = self.data[
                            self.current_step + self.window_size:
                            self.current_step + self.window_size + self.forecast_horizon
                            ]

            # Forecast future states
            try:
                predicted_future = self.forecast_future_states(current_state)

                # Calculate forecast error
                forecast_error = np.mean(np.abs(predicted_future - actual_future))

                # Enhanced reward function
                reward = -forecast_error  # Base reward is negative error

                # Action-based reward modification with momentum consideration
                price_trend = np.sign(predicted_future[-1] - predicted_future[0])
                volatility = np.std(predicted_future) / np.mean(predicted_future) if np.mean(
                    predicted_future) != 0 else 0

                if action == 1:  # Buy
                    if price_trend > 0:  # Predicted upward trend
                        reward += 0.2 * (1 + price_trend) * (1 - volatility)
                    else:
                        reward -= 0.2 * (1 + abs(price_trend)) * (1 + volatility)
                elif action == 2:  # Sell
                    if price_trend < 0:  # Predicted downward trend
                        reward += 0.2 * (1 + abs(price_trend)) * (1 - volatility)
                    else:
                        reward -= 0.2 * (1 + price_trend) * (1 + volatility)

                # Add reward for correct direction prediction
                actual_trend = np.sign(actual_future[-1] - current_price)
                if np.sign(price_trend) == np.sign(actual_trend):
                    reward += 0.1

            except Exception as e:
                print(f"Error in forecasting: {e}")
                reward = -1.0
                forecast_error = 1.0
        else:
            reward = 0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self.get_state() if not done else current_state

        info = {
            'forecast_error': forecast_error,
            'predicted_future': predicted_future,
            'current_price': current_price
        }

        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        return self.get_state()


class OptimizedDQNAgent:
    """
    Optimized Deep Q-Network Agent for faster training
    """

    def __init__(self, state_dim: int, action_dim: int, num_nodes: int, window_size: int = 20):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.window_size = window_size

        # Create T-GCN layers
        self.tgcn = OptimizedTemporalGraphConvLayer(1, 32, 16, num_nodes, dropout=0.2)

        # Calculate the flattened size after T-GCN
        flattened_size = window_size * num_nodes * 16

        # Q-Network with dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Target Network - make sure architecture matches the online network
        self.target_value_stream = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.target_advantage_stream = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Initialize target networks with same weights
        self.target_value_stream.load_state_dict(self.value_stream.state_dict())
        self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())

        # Set target networks to evaluation mode
        self.target_value_stream.eval()
        self.target_advantage_stream.eval()

        self.optimizer = optim.Adam(
            list(self.tgcn.parameters()) +
            list(self.value_stream.parameters()) +
            list(self.advantage_stream.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(10000)
        self.batch_size = 64

        # Exploration parameters with adaptive epsilon
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_adaptive = True

        # Q-learning parameters
        self.gamma = 0.99  # Increased discount factor
        self.tau = 0.01  # For soft target updates
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def get_state_dict(self):
        """Get the complete state of the agent"""
        return {
            'tgcn_state': copy.deepcopy(self.tgcn.state_dict()),
            'value_stream_state': copy.deepcopy(self.value_stream.state_dict()),
            'advantage_stream_state': copy.deepcopy(self.advantage_stream.state_dict()),
            'target_value_stream_state': copy.deepcopy(self.target_value_stream.state_dict()),
            'target_advantage_stream_state': copy.deepcopy(self.target_advantage_stream.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler_state': copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else None,
            'epsilon': self.epsilon,
        }

    def load_state_dict(self, state_dict):
        """Load the complete state of the agent"""
        self.tgcn.load_state_dict(state_dict['tgcn_state'])
        self.value_stream.load_state_dict(state_dict['value_stream_state'])
        self.advantage_stream.load_state_dict(state_dict['advantage_stream_state'])
        self.target_value_stream.load_state_dict(state_dict['target_value_stream_state'])
        self.target_advantage_stream.load_state_dict(state_dict['target_advantage_stream_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if self.scheduler and state_dict['scheduler_state']:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.epsilon = state_dict['epsilon']

    def _process_state_through_tgcn(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Process state through T-GCN and flatten for Q-network"""
        tgcn_output = self.tgcn(state_tensor)  # [batch, seq_len, num_nodes, hidden]
        flattened = tgcn_output.flatten(start_dim=1)  # [batch, seq_len * num_nodes * hidden]
        return flattened

    def get_q_values(self, state_tensor: torch.Tensor, target=False) -> torch.Tensor:
        """Get Q-values using dueling architecture"""
        features = self._process_state_through_tgcn(state_tensor)

        if target:
            value = self.target_value_stream(features)
            advantage = self.target_advantage_stream(features)
        else:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

        # Combine value and advantage streams
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state: np.ndarray, eval_mode=False) -> int:
        """
        Choose action using epsilon-greedy policy
        """
        if not eval_mode and np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        # Prepare state tensor with correct dimensions [batch, seq, nodes, features]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(-1)  # [1, window_size, 1, 1]

        try:
            # Process through T-GCN then Q-network
            q_values = self.get_q_values(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
        except Exception as e:
            print(f"Error in action selection: {e}")
            return random.randrange(self.action_dim)

    def replay(self):
        """
        Train the Q-network using prioritized experience replay
        """
        if len(self.memory) < self.batch_size:
            return 0  # Return zero loss if not enough samples

        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        if len(states) == 0:
            return 0

        # Convert to tensors
        states = torch.FloatTensor(states).unsqueeze(-1)  # [batch_size, window_size, 1, 1]
        next_states = torch.FloatTensor(next_states).unsqueeze(-1)  # [batch_size, window_size, 1, 1]
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        weights = torch.FloatTensor(weights)

        # Compute current Q values
        current_q_values = self.get_q_values(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values from target network
        with torch.no_grad():
            next_q_values = self.get_q_values(next_states, target=True)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_values = next_q_values.gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        # Compute loss with importance sampling weights
        loss = (weights.unsqueeze(1) * self.loss_fn(current_q_values, target_q_values)).mean()

        # Calculate priorities for replay buffer
        priorities = (torch.abs(current_q_values - target_q_values).detach().numpy() + 1e-6).flatten()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.tgcn.parameters()) +
            list(self.value_stream.parameters()) +
            list(self.advantage_stream.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, priorities)

        # Update target networks with soft updates
        self.soft_update_target_networks()

        # Decay epsilon
        if self.epsilon_adaptive:
            # Adaptive epsilon decay based on performance
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def soft_update_target_networks(self):
        """Soft update target network weights"""
        for target_param, param in zip(self.target_value_stream.parameters(), self.value_stream.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_advantage_stream.parameters(), self.advantage_stream.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_target_network(self):
        """Hard update target network weights"""
        self.target_value_stream.load_state_dict(self.value_stream.state_dict())
        self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())

        # Set target networks to evaluation mode
        self.target_value_stream.eval()
        self.target_advantage_stream.eval()


class OptimizedRLTSPFramework:
    """
    Optimized RLTSP Framework for faster training
    """

    def __init__(self, data: np.ndarray, window_size: int = 10, forecast_horizon: int = 5):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

        self.env = OptimizedPredictiveEnvironment(data, window_size, forecast_horizon)
        self.agent = OptimizedDQNAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            num_nodes=self.env.num_nodes,
            window_size=window_size
        )

        self.training_history = {
            'rewards': [],
            'forecast_errors': [],
            'epsilon': [],
            'losses': [],
            'actions': {0: [], 1: [], 2: []},
            'portfolio_value': [10000]
        }

        # For model checkpointing
        self.best_reward = -float('inf')
        self.best_model = None
        self.patience = 20
        self.patience_counter = 0

    def get_state_dict(self):
        """Get the complete state of the framework"""
        return {
            'env_state': self.env.get_model_state(),
            'agent_state': self.agent.get_state_dict(),
            'training_history': copy.deepcopy(self.training_history),
            'best_reward': self.best_reward,
            'patience_counter': self.patience_counter
        }

    def load_state_dict(self, state_dict):
        """Load the complete state of the framework"""
        self.env.set_model_state(state_dict['env_state'])
        self.agent.load_state_dict(state_dict['agent_state'])
        self.training_history = state_dict['training_history']
        self.best_reward = state_dict['best_reward']
        self.patience_counter = state_dict['patience_counter']

    def train_forecaster(self, epochs: int = 50, batch_size: int = 32):
        """Pre-train the forecasting component with batch training"""
        print("Training forecasting component...")

        # Create training dataset
        X_train, y_train = [], []
        for i in range(0, len(self.env.data) - self.env.window_size - self.env.forecast_horizon, 1):
            X = self.env.data[i:i + self.env.window_size]
            y = self.env.data[i + self.env.window_size]
            X_train.append(X.reshape(-1, 1))
            y_train.append(y)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        num_batches = int(np.ceil(len(X_train) / batch_size))

        for epoch in range(epochs):
            total_loss = 0

            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                X_tensor = torch.FloatTensor(X_batch).unsqueeze(-1)  # [batch_size, window_size, 1, 1]
                y_tensor = torch.FloatTensor(y_batch)

                loss = self.env.train_forecaster(X_tensor, y_tensor)
                total_loss += loss

            avg_loss = total_loss / num_batches
            if self.env.scheduler:
                self.env.scheduler.step(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

    def train(self, episodes: int = 200, update_target_freq: int = 10):
        """Train the complete optimized RLTSP framework"""
        print("Training Optimized RLTSP framework...")
        start_time = time.time()

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            total_forecast_error = 0
            steps = 0
            episode_loss = 0
            action_counts = {0: 0, 1: 0, 2: 0}  # Track actions in this episode

            # Initialize portfolio for this episode
            cash = 10000
            holdings = 0
            portfolio_value = cash

            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)

                # Update portfolio based on action
                current_price = info.get('current_price', 0)
                if action == 1 and cash > current_price:  # Buy
                    shares_to_buy = cash // current_price
                    cash -= shares_to_buy * current_price
                    holdings += shares_to_buy
                elif action == 2 and holdings > 0:  # Sell
                    cash += holdings * current_price
                    holdings = 0

                # Calculate portfolio value
                portfolio_value = cash + holdings * current_price

                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                total_forecast_error += info.get('forecast_error', 0)
                action_counts[action] += 1
                steps += 1

                # Train agent
                loss = self.agent.replay()
                episode_loss += loss if loss else 0

                if done:
                    break

            # Update target network periodically
            if episode % update_target_freq == 0:
                self.agent.update_target_network()

            # Record training history
            avg_reward = total_reward / steps if steps > 0 else 0
            avg_error = total_forecast_error / steps if steps > 0 else 0
            avg_loss = episode_loss / steps if steps > 0 else 0

            self.training_history['rewards'].append(avg_reward)
            self.training_history['forecast_errors'].append(avg_error)
            self.training_history['losses'].append(avg_loss)
            self.training_history['epsilon'].append(self.agent.epsilon)
            self.training_history['portfolio_value'].append(portfolio_value)

            # Track action distribution
            for action in action_counts:
                self.training_history['actions'][action].append(action_counts[action] / steps if steps > 0 else 0)

            # Early stopping and model checkpointing
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.best_model = self.get_state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping at episode {episode}")
                break

            if episode % 20 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, "
                      f"Avg Forecast Error: {avg_error:.6f}, Avg Loss: {avg_loss:.6f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}, Portfolio: ${portfolio_value:.2f}, "
                      f"Time: {elapsed_time:.2f}s")

        # Restore best model
        if self.best_model:
            self.load_state_dict(self.best_model)

    def predict(self, input_data: np.ndarray, steps: int = 5) -> np.ndarray:
        """Make predictions using trained RLTSP model"""
        self.agent.value_stream.eval()
        self.agent.advantage_stream.eval()
        self.env.tgcn.eval()
        self.env.forecast_head.eval()

        predictions = []
        action_sequence = []
        current_state = input_data[-self.env.window_size:].reshape(-1, 1)  # Ensure 2D shape

        # Initialize portfolio for simulation
        cash = 10000
        holdings = 0
        portfolio_values = [cash]

        with torch.no_grad():
            for i in range(steps):
                # Get forecast from environment
                forecast = self.env.forecast_future_states(current_state, num_steps=1)

                # Get action from agent
                action = self.agent.act(current_state, eval_mode=True)
                action_sequence.append(action)

                # Simulate trading action
                current_price = forecast[0] if len(forecast) > 0 else current_state[-1, 0]

                if action == 1 and cash > current_price:  # Buy
                    shares_to_buy = cash // current_price
                    cash -= shares_to_buy * current_price
                    holdings += shares_to_buy
                elif action == 2 and holdings > 0:  # Sell
                    cash += holdings * current_price
                    holdings = 0

                # Calculate portfolio value
                portfolio_value = cash + holdings * current_price
                portfolio_values.append(portfolio_value)

                # Store prediction
                predictions.append(float(forecast[0]))

                # Update state for next prediction
                current_state = np.append(current_state[1:], forecast[0]).reshape(-1, 1)

        return np.array(predictions), np.array(action_sequence), np.array(portfolio_values)

    def evaluate(self, test_data: np.ndarray) -> Dict:
        """Evaluate the model on test data"""
        predictions, actions, portfolio_values = self.predict(test_data, steps=len(test_data) - self.env.window_size)

        # Calculate metrics
        actual = test_data[self.env.window_size:]
        mae = np.mean(np.abs(predictions - actual))
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)

        # Calculate directional accuracy
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actual))
        directional_accuracy = np.mean(pred_direction == actual_direction)

        # Calculate portfolio return
        initial_portfolio = portfolio_values[0]
        final_portfolio = portfolio_values[-1]
        portfolio_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'portfolio_return': portfolio_return,
            'predictions': predictions,
            'actions': actions,
            'portfolio_values': portfolio_values
        }


if __name__ == "__main__":
    # Load data using the data_loader module
    print("Loading training data...")
    data = load_data()
    
    # Create and train the RLTSP framework
    rltsp = OptimizedRLTSPFramework(data, window_size=20, forecast_horizon=5)
    
    # Pre-train the forecaster
    print("Pre-training forecaster...")
    rltsp.train_forecaster(epochs=10)
    
    # Train the complete framework
    print("Training RLTSP...")
    rewards = rltsp.train(episodes=100)
    
    # Evaluate on test data
    print("Loading test data...")
    test_data = load_test_data()
    results = rltsp.evaluate(test_data)
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    print(f"Test MAE: {results['mae']:.4f}")
    print(f"Test RMSE: {results['rmse']:.4f}")
    print(f"Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"Portfolio Return: {results['portfolio_return']:.2f}%")
    
    print("\nOptimized RLTSP Framework training and evaluation completed!")