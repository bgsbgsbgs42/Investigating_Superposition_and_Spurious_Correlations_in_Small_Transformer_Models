# bitcoin_analysis.py
import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any

class BTCDataProcessor:
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        
    def get_btc_data(self) -> pd.DataFrame:
        """Download and process BTC data"""
        btc = yf.download('BTC-USD', start='2016-02-10', end='2024-02-10', interval='1mo')
        
        # Technical features
        btc['SMA_3'] = btc['Close'].rolling(window=3).mean()
        btc['RSI'] = self._calculate_rsi(btc['Close'])
        btc['Volatility'] = btc['Close'].rolling(window=3).std()
        btc['Volume_MA'] = btc['Volume'].rolling(window=3).mean()
        btc['Price_Change'] = btc['Close'].pct_change()
        
        # Market cycle features
        btc['DayOfYear'] = btc.index.dayofyear
        btc['MonthEnd'] = btc.index.is_month_end.astype(int)
        btc['QuarterEnd'] = btc.index.is_quarter_end.astype(int)
        btc['YearEnd'] = btc.index.is_year_end.astype(int)
        
        return btc.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for training"""
        features = [
            'SMA_3', 'RSI', 'Volatility', 'Volume_MA', 'Price_Change',
            'DayOfYear', 'MonthEnd', 'QuarterEnd', 'YearEnd'
        ]
        
        X, y = [], []
        for i in range(self.lookback, len(data)):
            feature_sequence = data[features].iloc[i-self.lookback:i].values
            X.append(feature_sequence)
            # Label: 1 if price increases in next period
            y.append(data['Close'].iloc[i] > data['Close'].iloc[i-1])
            
        return torch.FloatTensor(X), torch.FloatTensor(y)

class BTCModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=2
        )
        self.output = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.transformer(x)
        return torch.sigmoid(self.output(x.mean(dim=1))))

class BTCDetailedAnalyzer:
    def __init__(self, model: nn.Module, features: torch.Tensor, labels: torch.Tensor):
        self.model = model
        self.features = features
        self.labels = labels
        
    def analyze_volatility_regimes(self) -> Dict[str, Dict]:
        """Analyze model behavior in different volatility regimes"""
        volatility = self.features[:, :, 2].mean(dim=1)  # Volatility feature
        
        # Define regimes
        low_vol = torch.quantile(volatility, 0.33)
        high_vol = torch.quantile(volatility, 0.66)
        
        regimes = {
            'low_vol': volatility <= low_vol,
            'med_vol': (volatility > low_vol) & (volatility <= high_vol),
            'high_vol': volatility > high_vol
        }
        
        regime_metrics = {}
        with torch.no_grad():
            predictions = self.model(self.features)
            
            for regime_name, mask in regimes.items():
                regime_metrics[regime_name] = {
                    'accuracy': ((predictions[mask] > 0.5) == self.labels[mask]).float().mean().item(),
                    'confidence': torch.abs(predictions[mask] - 0.5).mean().item(),
                    'feature_importance': self._analyze_feature_importance(mask)
                }
                
        return regime_metrics
    
    def _analyze_feature_importance(self, mask: torch.Tensor) -> list:
        """Analyze feature importance for a subset of data"""
        with torch.no_grad():
            base_pred = self.model(self.features[mask])
            importance = []
            
            for i in range(self.features.shape[-1]):
                modified = self.features[mask].clone()
                modified[:, :, i] = 0
                new_pred = self.model(modified)
                importance.append(torch.abs(base_pred - new_pred).mean().item())
                
            return importance
    
    def analyze_market_cycles(self) -> Dict[str, Any]:
        """Analyze dependence on market cycles"""
        with torch.no_grad():
            predictions = self.model(self.features)
        
        # Convert to numpy for analysis
        preds_np = predictions.numpy()
        features_np = self.features.numpy()
        
        # Analyze cycle dependence
        cycle_metrics = {
            'yearly': np.corrcoef(features_np[:, :, 5].mean(axis=1), preds_np)[0,1],
            'monthly': np.corrcoef(features_np[:, :, 6].mean(axis=1), preds_np)[0,1],
            'quarterly': np.corrcoef(features_np[:, :, 7].mean(axis=1), preds_np)[0,1]
        }
        
        # Analyze technical vs cycle importance
        technical_importance = np.mean([
            np.abs(np.corrcoef(features_np[:, :, i].mean(axis=1), preds_np)[0,1])
            for i in range(5)
        ])
        
        cycle_importance = np.mean([
            np.abs(np.corrcoef(features_np[:, :, i].mean(axis=1), preds_np)[0,1])
            for i in range(5, 9)
        ])
        
        return {
            'cycle_correlations': cycle_metrics,
            'feature_importance': {
                'technical': technical_importance,
                'market_cycle': cycle_importance
            }
        }

def train_btc_model() -> Dict[str, Any]:
    """Train BTC model and return analysis results"""
    # Process data
    processor = BTCDataProcessor()
    btc_data = processor.get_btc_data()
    features, labels = processor.prepare_training_data(btc_data)
    
    # Initialize model
    model = BTCModel(input_dim=features.shape[-1])
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Analyze results
    analyzer = BTCDetailedAnalyzer(model, features, labels)
    volatility_analysis = analyzer.analyze_volatility_regimes()
    cycle_analysis = analyzer.analyze_market_cycles()
    
    return {
        'model': model,
        'features': features,
        'labels': labels,
        'volatility_analysis': volatility_analysis,
        'cycle_analysis': cycle_analysis
    }

if __name__ == "__main__":
    results = train_btc_model()
    
    print("\nVolatility Regime Analysis:")
    for regime, metrics in results['volatility_analysis'].items():
        print(f"\n{regime}:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Confidence: {metrics['confidence']:.3f}")
        
    print("\nMarket Cycle Analysis:")
    for cycle, corr in results['cycle_analysis']['cycle_correlations'].items():
        print(f"{cycle}: {corr:.3f}")
    
    print("\nFeature Importance:")
    for feature_type, importance in results['cycle_analysis']['feature_importance'].items():
        print(f"{feature_type}: {importance:.3f}")