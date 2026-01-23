"""
Deep Learning models for probabilistic forecasting.

Models implemented:
- LSTMQuantileRegression: LSTM with quantile outputs
- TransformerQuantileRegression: Transformer with quantile outputs
- DeepEnsemble: Multiple neural networks for uncertainty
- MCDropoutLSTM: Bayesian approximation via MC Dropout

All models include conformal calibration for coverage guarantees.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple
import logging

from .base import BaseDeepLearningForecaster, PredictionResult

logger = logging.getLogger(__name__)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    L_q(y, y_hat) = q * max(y - y_hat, 0) + (1-q) * max(y_hat - y, 0)
    """
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : torch.Tensor, shape (batch, n_quantiles)
        targets : torch.Tensor, shape (batch,)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses, dim=1))


# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class LSTMQuantileNet(nn.Module):
    """LSTM network for quantile regression."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        quantiles: List[float] = [0.025, 0.5, 0.975]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.quantiles = quantiles
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(quantiles))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        return self.fc(out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerQuantileNet(nn.Module):
    """Transformer network for quantile regression."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = [0.025, 0.5, 0.975]
    ):
        super().__init__()
        self.quantiles = quantiles
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, len(quantiles))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)


class SimpleNet(nn.Module):
    """Simple feedforward network for ensemble."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)


class MCDropoutLSTMNet(nn.Module):
    """LSTM with MC Dropout for Bayesian uncertainty."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# =============================================================================
# LSTM QUANTILE REGRESSION
# =============================================================================

class LSTMQuantileRegression(BaseDeepLearningForecaster):
    """
    LSTM-based Quantile Regression for Probabilistic Forecasting.
    
    Uses a 2-layer LSTM with quantile outputs followed by
    conformal calibration for coverage guarantees.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(
            alpha=alpha,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
        self.model: Optional[LSTMQuantileNet] = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "LSTMQuantileRegression":
        """Train LSTM and calibrate prediction intervals."""
        logger.info("Training LSTM Quantile Regression...")
        logger.info(f"Architecture: {self.num_layers} layers, hidden_size={self.hidden_size}")
        
        # Set random seed
        torch.manual_seed(self.random_state)
        
        # Normalize features
        X_train_norm = self._normalize(X_train, fit=True)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = LSTMQuantileNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            quantiles=self.quantiles
        ).to(self.device)
        
        # Loss and optimizer
        criterion = QuantileLoss(self.quantiles)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        # Calibration
        logger.info("Applying conformal calibration...")
        self._calibrate(X_cal, y_cal)
        
        self._is_fitted = True
        return self
    
    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """Generate raw (uncalibrated) predictions."""
        self.model.eval()
        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return PredictionResult(
            point=predictions[:, 1],  # Median
            lower=predictions[:, 0],
            upper=predictions[:, 2]
        )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate calibrated predictions."""
        self._check_is_fitted()
        
        raw = self._predict_raw(X)
        
        return PredictionResult(
            point=raw.point,
            lower=raw.lower - self.calibration_adjustment,
            upper=raw.upper + self.calibration_adjustment
        )


# =============================================================================
# TRANSFORMER QUANTILE REGRESSION
# =============================================================================

class TransformerQuantileRegression(BaseDeepLearningForecaster):
    """
    Transformer-based Quantile Regression for Probabilistic Forecasting.
    
    Uses self-attention mechanism for capturing temporal dependencies.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(
            alpha=alpha,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
        self.model: Optional[TransformerQuantileNet] = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "TransformerQuantileRegression":
        """Train Transformer and calibrate prediction intervals."""
        logger.info("Training Transformer Quantile Regression...")
        logger.info(f"Architecture: {self.num_layers} layers, d_model={self.d_model}, heads={self.nhead}")
        
        torch.manual_seed(self.random_state)
        
        X_train_norm = self._normalize(X_train, fit=True)
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_size = X_train.shape[2]
        self.model = TransformerQuantileNet(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            quantiles=self.quantiles
        ).to(self.device)
        
        criterion = QuantileLoss(self.quantiles)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Applying conformal calibration...")
        self._calibrate(X_cal, y_cal)
        
        self._is_fitted = True
        return self
    
    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """Generate raw predictions."""
        self.model.eval()
        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return PredictionResult(
            point=predictions[:, 1],
            lower=predictions[:, 0],
            upper=predictions[:, 2]
        )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate calibrated predictions."""
        self._check_is_fitted()
        raw = self._predict_raw(X)
        
        return PredictionResult(
            point=raw.point,
            lower=raw.lower - self.calibration_adjustment,
            upper=raw.upper + self.calibration_adjustment
        )


# =============================================================================
# DEEP ENSEMBLE
# =============================================================================

class DeepEnsemble(BaseDeepLearningForecaster):
    """
    Deep Ensemble for Uncertainty Quantification.
    
    Trains multiple neural networks with different random initializations.
    Uncertainty is estimated from the spread of predictions.
    
    Reference: Lakshminarayanan et al. (2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        n_ensemble: int = 5,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(
            alpha=alpha,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device
        )
        self.n_ensemble = n_ensemble
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.models: List[SimpleNet] = []
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "DeepEnsemble":
        """Train ensemble of neural networks."""
        logger.info(f"Training Deep Ensemble ({self.n_ensemble} members)...")
        
        X_train_norm = self._normalize(X_train, fit=True)
        X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
        input_size = X_train_flat.shape[1]
        
        X_tensor = torch.FloatTensor(X_train_flat).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        for i in range(self.n_ensemble):
            logger.info(f"Training member {i + 1}/{self.n_ensemble}...")
            
            torch.manual_seed(self.random_state + i)
            
            model = SimpleNet(input_size, self.hidden_size).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            model.train()
            for epoch in range(self.epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.models.append(model)
        
        logger.info("Applying conformal calibration...")
        self._calibrate(X_cal, y_cal)
        
        self._is_fitted = True
        return self
    
    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """Generate raw ensemble predictions."""
        X_norm = self._normalize(X)
        X_flat = X_norm.reshape(X_norm.shape[0], -1)
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        
        all_predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        return PredictionResult(
            point=np.mean(all_predictions, axis=0),
            lower=np.percentile(all_predictions, (self.alpha / 2) * 100, axis=0),
            upper=np.percentile(all_predictions, (1 - self.alpha / 2) * 100, axis=0)
        )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate calibrated predictions."""
        self._check_is_fitted()
        raw = self._predict_raw(X)
        
        return PredictionResult(
            point=raw.point,
            lower=raw.lower - self.calibration_adjustment,
            upper=raw.upper + self.calibration_adjustment
        )


# =============================================================================
# MC DROPOUT LSTM
# =============================================================================

class MCDropoutLSTM(BaseDeepLearningForecaster):
    """
    Monte Carlo Dropout LSTM for Bayesian Uncertainty Estimation.
    
    Uses dropout at inference time to approximate Bayesian inference.
    
    Reference: Gal & Ghahramani (2016)
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        n_mc_samples: int = 100,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(
            alpha=alpha,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples
        
        self.model: Optional[MCDropoutLSTMNet] = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "MCDropoutLSTM":
        """Train MC Dropout LSTM."""
        logger.info(f"Training MC Dropout LSTM ({self.n_mc_samples} MC samples at inference)...")
        
        torch.manual_seed(self.random_state)
        
        X_train_norm = self._normalize(X_train, fit=True)
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_size = X_train.shape[2]
        self.model = MCDropoutLSTMNet(
            input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        
        logger.info("Applying conformal calibration...")
        self._calibrate(X_cal, y_cal)
        
        self._is_fitted = True
        return self
    
    def _mc_predict(self, X: np.ndarray) -> np.ndarray:
        """Generate MC samples with dropout active."""
        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        # Keep dropout active
        self.model.train()
        
        all_predictions = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred = self.model(X_tensor).cpu().numpy()
                all_predictions.append(pred)
        
        return np.array(all_predictions)
    
    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """Generate raw MC predictions."""
        all_predictions = self._mc_predict(X)
        
        return PredictionResult(
            point=np.mean(all_predictions, axis=0),
            lower=np.percentile(all_predictions, (self.alpha / 2) * 100, axis=0),
            upper=np.percentile(all_predictions, (1 - self.alpha / 2) * 100, axis=0)
        )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate calibrated predictions."""
        self._check_is_fitted()
        raw = self._predict_raw(X)
        
        return PredictionResult(
            point=raw.point,
            lower=raw.lower - self.calibration_adjustment,
            upper=raw.upper + self.calibration_adjustment
        )


# =============================================================================
# TEMPORAL FUSION TRANSFORMER (TFT)
# =============================================================================

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from TFT paper.
    
    Applies ELU activation, dropout, linear layer, and gating mechanism.
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        
        # Skip connection
        if input_size != hidden_size:
            self.skip = nn.Linear(input_size, hidden_size)
        else:
            self.skip = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for skip connection
        residual = x if self.skip is None else self.skip(x)
        
        # Main path
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        # Gating mechanism (GLU-style)
        gate = torch.sigmoid(self.gate(out))
        out = out * gate
        
        # Add residual and normalize
        out = self.layernorm(out + residual)
        return out


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network from TFT.
    
    Learns to select relevant features using softmax attention.
    """
    
    def __init__(self, input_size: int, num_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Feature-specific GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, dropout)
            for _ in range(num_features)
        ])
        
        # Variable selection weights
        self.softmax_grn = GatedResidualNetwork(input_size * num_features, num_features, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_features, input_size)
        Returns: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, num_features, input_size = x.shape
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(batch_size * seq_len, num_features, input_size)
        
        # Process each feature through its GRN
        processed_features = []
        for i, grn in enumerate(self.feature_grns):
            feat = grn(x_flat[:, i, :])
            processed_features.append(feat)
        processed_features = torch.stack(processed_features, dim=1)  # (batch*seq, num_feat, hidden)
        
        # Compute selection weights
        concatenated = x_flat.view(batch_size * seq_len, -1)  # (batch*seq, num_feat * input_size)
        weights = self.softmax_grn(concatenated)  # (batch*seq, num_features)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # (batch*seq, num_features, 1)
        
        # Apply weights
        selected = (processed_features * weights).sum(dim=1)  # (batch*seq, hidden)
        
        # Reshape back
        selected = selected.view(batch_size, seq_len, self.hidden_size)
        return selected


class TemporalFusionTransformerNet(nn.Module):
    """
    Temporal Fusion Transformer (TFT) network for quantile forecasting.
    
    Simplified implementation with key TFT components:
    - Variable Selection Networks
    - Gated Residual Networks
    - Multi-head attention
    - Quantile outputs
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = [0.025, 0.5, 0.975]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        
        # Input processing: treat each time step as a feature
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Multi-head attention for temporal fusion
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # GRNs for processing
        self.grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=100)
        
        # Quantile output heads
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(quantiles))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        Returns: (batch, num_quantiles)
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq_len, hidden)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Multi-head attention layers with GRN
        for attention, grn in zip(self.attention_layers, self.grns):
            # Self-attention
            attn_out, _ = attention(lstm_out, lstm_out, lstm_out)
            
            # Apply GRN with residual
            lstm_out = grn(attn_out + lstm_out)
        
        # Global pooling
        pooled = lstm_out.mean(dim=1)  # (batch, hidden)
        
        # Quantile predictions
        quantiles = self.quantile_head(pooled)  # (batch, num_quantiles)
        
        return quantiles


class TemporalFusionTransformer(BaseDeepLearningForecaster):
    """
    Temporal Fusion Transformer for multi-horizon probabilistic forecasting.

    TFT combines:
    - Variable selection for interpretability
    - LSTM for local processing
    - Multi-head attention for long-range dependencies
    - Quantile outputs for uncertainty

    Reference: Lim et al. (2021) "Temporal Fusion Transformers for
    Interpretable Multi-horizon Time Series Forecasting"
    """

    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(
            alpha=alpha,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
        self.model: Optional[TemporalFusionTransformerNet] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "TemporalFusionTransformer":
        """Train TFT and calibrate prediction intervals."""
        logger.info("Training Temporal Fusion Transformer...")
        logger.info(f"Architecture: {self.num_layers} layers, {self.num_heads} heads, hidden_size={self.hidden_size}")

        # Set random seed
        torch.manual_seed(self.random_state)

        # Normalize features
        X_train_norm = self._normalize(X_train, fit=True)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_size = X_train.shape[2]
        self.model = TemporalFusionTransformerNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            quantiles=self.quantiles
        ).to(self.device)

        # Loss and optimizer
        criterion = QuantileLoss(self.quantiles)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Calibration
        logger.info("Applying conformal calibration...")
        self._calibrate(X_cal, y_cal)

        self._is_fitted = True
        return self

    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """Generate raw (uncalibrated) predictions."""
        self.model.eval()
        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return PredictionResult(
            point=predictions[:, 1],  # Median
            lower=predictions[:, 0],
            upper=predictions[:, 2]
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate calibrated predictions."""
        self._check_is_fitted()

        raw = self._predict_raw(X)

        return PredictionResult(
            point=raw.point,
            lower=raw.lower - self.calibration_adjustment,
            upper=raw.upper + self.calibration_adjustment
        )


# =============================================================================
# SPO/END-TO-END BASELINE
# =============================================================================

class SPOLoss(nn.Module):
    """
    Smart Predict-then-Optimize (SPO) loss function.

    Directly optimizes the downstream newsvendor decision cost.
    This is a decision-focused learning approach that trains models
    to minimize the actual inventory cost rather than prediction error.

    Reference: Elmachtoub & Grigas (2017) "Smart 'Predict, then Optimize'"
    """

    def __init__(
        self,
        ordering_cost: float = 10.0,
        holding_cost: float = 2.0,
        stockout_cost: float = 50.0,
        beta: float = 0.90,
        n_samples: int = 100
    ):
        super().__init__()
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.beta = beta
        self.n_samples = n_samples

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SPO loss based on newsvendor cost.

        Parameters
        ----------
        predictions : torch.Tensor, shape (batch, 3)
            Predicted quantiles [lower, median, upper]
        targets : torch.Tensor, shape (batch,)
            Actual demand values

        Returns
        -------
        torch.Tensor
            Mean newsvendor cost (decision loss)
        """
        batch_size = predictions.shape[0]

        # Extract quantile predictions
        lower_pred = predictions[:, 0]
        point_pred = predictions[:, 1]
        upper_pred = predictions[:, 2]

        # Compute order quantities using predicted distribution
        # For simplicity, use the median as order quantity
        # In practice, we could sample from the predicted interval
        order_qty = point_pred

        # Compute newsvendor loss for each sample
        overage = torch.clamp(order_qty - targets, min=0)
        underage = torch.clamp(targets - order_qty, min=0)

        cost = (
            self.ordering_cost * order_qty +
            self.holding_cost * overage +
            self.stockout_cost * underage
        )

        # Return mean cost as the loss
        return cost.mean()


class SPONet(nn.Module):
    """
    Neural network for SPO/End-to-End learning.

    Outputs quantile predictions that will be optimized
    to minimize downstream newsvendor costs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        quantiles: List[float] = [0.025, 0.5, 0.975]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.quantiles = quantiles

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers for quantile predictions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, features)

        Returns
        -------
        torch.Tensor, shape (batch, n_quantiles)
            Quantile predictions
        """
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        return self.fc(out)


class SPOEndToEnd(BaseDeepLearningForecaster):
    """
    SPO/End-to-End Baseline for Inventory Optimization.

    This model directly optimizes the downstream decision cost (newsvendor loss)
    rather than prediction accuracy. It represents the critical competitor to
    traditional predict-then-optimize approaches.

    Key differences from traditional approaches:
    - Loss function: Newsvendor cost instead of MSE/quantile loss
    - Objective: Minimize inventory costs directly
    - Decision-focused: Learns predictions that lead to better decisions

    References:
    - Elmachtoub & Grigas (2017) "Smart 'Predict, then Optimize'"
    - Donti et al. (2017) "Task-based End-to-end Model Learning"
    """

    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        ordering_cost: float = 10.0,
        holding_cost: float = 2.0,
        stockout_cost: float = 50.0,
        beta: float = 0.90,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(
            alpha=alpha,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Cost parameters for newsvendor problem
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.beta = beta

        self.quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
        self.model: Optional[SPONet] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "SPOEndToEnd":
        """Train SPO model by minimizing newsvendor costs."""
        logger.info("Training SPO/End-to-End Baseline...")
        logger.info(f"Decision-focused learning with newsvendor costs")
        logger.info(f"Architecture: {self.num_layers} layers, hidden_size={self.hidden_size}")

        # Set random seed
        torch.manual_seed(self.random_state)

        # Normalize features
        X_train_norm = self._normalize(X_train, fit=True)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_size = X_train.shape[2]
        self.model = SPONet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            quantiles=self.quantiles
        ).to(self.device)

        # SPO loss function (decision-focused)
        criterion = SPOLoss(
            ordering_cost=self.ordering_cost,
            holding_cost=self.holding_cost,
            stockout_cost=self.stockout_cost,
            beta=self.beta
        )

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Cost: ${avg_loss:.2f}")

        # Calibration
        logger.info("Applying conformal calibration...")
        self._calibrate(X_cal, y_cal)

        self._is_fitted = True
        return self

    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """Generate raw (uncalibrated) predictions."""
        self.model.eval()
        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return PredictionResult(
            point=predictions[:, 1],  # Median
            lower=predictions[:, 0],
            upper=predictions[:, 2]
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate calibrated predictions."""
        self._check_is_fitted()

        raw = self._predict_raw(X)

        return PredictionResult(
            point=raw.point,
            lower=raw.lower - self.calibration_adjustment,
            upper=raw.upper + self.calibration_adjustment
        )
