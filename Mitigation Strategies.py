import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance
from sklearn.decomposition import FastICA
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data Generation
def generate_biased_data(n_samples=1000, n_features=5, feature_dim=10, bias_strength=0.8):
    """Generate synthetic data with intentional spurious correlations"""
    # Generate core features
    features = np.random.normal(0, 1, (n_samples, n_features * feature_dim))
    
    # Generate labels based on core features
    labels = (np.sum(features[:, :feature_dim*3]**2, axis=1) > feature_dim*1.5).astype(float)
    
    # Add biased feature that correlates with labels
    bias_feature = np.random.normal(0, 1, (n_samples, feature_dim))
    bias_feature[labels == 1] += 2  # Make bias feature predictive for positive class
    
    # Mix features
    features = np.concatenate([features, bias_feature], axis=1)
    
    # Add some noise to labels
    labels = np.where(np.random.random(n_samples) < bias_strength, 
                     (np.sum(bias_feature, axis=1) > 0, 
                     labels)
    
    return torch.FloatTensor(features).to(device), torch.FloatTensor(labels).to(device)

# 2. Model Definition
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.input_proj(x.unsqueeze(1))  # Add sequence dimension
        x = self.transformer(x)
        return torch.sigmoid(self.output_layer(x.mean(dim=1))).squeeze()

# 3. Enhanced Detector
class EnhancedDetector:
    def __init__(self, model: nn.Module, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.activations = {}
        
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        return hooks
    
    def detect_spurious_correlations(self, 
                                   features: torch.Tensor,
                                   labels: torch.Tensor,
                                   feature_groups: Dict[str, slice]) -> Dict[str, Dict[str, float]]:
        """Detect spurious correlations with enhanced metrics"""
        hooks = self._register_hooks()
        with torch.no_grad():
            _ = self.model(features)
        
        scores = {}
        for group_name, group_slice in feature_groups.items():
            group_features = features[:, group_slice]
            
            metrics = {
                'predictive_power': self._calculate_predictive_power(group_features, labels),
                'counterfactual_impact': self._measure_counterfactual_impact(
                    group_features, features, labels),
                'distribution_shift': self._measure_distribution_shift(group_features, labels)
            }
            scores[group_name] = metrics
            
        for hook in hooks:
            hook.remove()
            
        return scores
    
    def _calculate_predictive_power(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate mutual information between features and labels"""
        flat_features = features.reshape(-1, features.shape[-1]).cpu().numpy()
        mi_scores = [mutual_info_score(flat_features[:, i], labels.cpu().numpy()) 
                    for i in range(flat_features.shape[1])]
        return float(np.mean(mi_scores))
    
    def _measure_counterfactual_impact(self, group_features, full_features, labels) -> float:
        """Measure impact of permuting feature group on predictions"""
        with torch.no_grad():
            orig_pred = self.model(full_features)
            
            # Create counterfactual by permuting the group features
            perm_idx = torch.randperm(len(group_features))
            cf_features = full_features.clone()
            cf_features[:, group_features.shape[1]:] = group_features[perm_idx]
            
            cf_pred = self.model(cf_features)
            return torch.mean(torch.abs(orig_pred - cf_pred)).item()
    
    def _measure_distribution_shift(self, features, labels) -> float:
        """Calculate Wasserstein distance between positive and negative classes"""
        pos = features[labels == 1].cpu().numpy()
        neg = features[labels == 0].cpu().numpy()
        
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
            
        distances = []
        for i in range(features.shape[1]):
            dist = wasserstein_distance(pos[:, i], neg[:, i])
            distances.append(dist)
        return float(np.mean(distances))

# 4. Enhanced Mitigator
class EnhancedMitigator:
    def __init__(self, model: nn.Module, detector: EnhancedDetector):
        self.model = model
        self.detector = detector
        
    def mitigate_spurious_correlations(self, features, labels, feature_groups):
        """Apply targeted mitigation based on detection scores"""
        scores = self.detector.detect_spurious_correlations(features, labels, feature_groups)
        
        for group_name, metrics in scores.items():
            if any(v > self.detector.threshold for v in metrics.values()):
                print(f"Mitigating spurious correlations in {group_name}")
                self._apply_feature_mitigation(feature_groups[group_name])
    
    def _apply_feature_mitigation(self, feature_slice):
        """Apply regularization to reduce spurious feature influence"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                with torch.no_grad():
                    # Reduce weights for problematic features
                    param.data[:, feature_slice] *= 0.8
                    
                    # Add small noise to break exact correlations
                    param.data[:, feature_slice] += torch.randn_like(param.data[:, feature_slice]) * 0.01

# 5. Training Function
def train_model(model, features, labels, n_epochs=100, batch_size=32):
    """Train model with early stopping"""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    return model

# 6. Evaluation Functions
def evaluate_model(model, features, labels):
    """Evaluate model accuracy"""
    with torch.no_grad():
        predictions = model(features)
        accuracy = ((predictions > 0.5).float() == labels).float().mean()
    return accuracy.item()

def test_generalization(model, features, labels):
    """Test generalization by permuting spurious features"""
    with torch.no_grad():
        # Create test set with permuted spurious features
        test_features = features.clone()
        test_features[:, -10:] = test_features[torch.randperm(len(features)), -10:]
        predictions = model(test_features)
        accuracy = ((predictions > 0.5).float() == labels).float().mean()
    return accuracy.item()

# 7. Main Execution
if __name__ == "__main__":
    # Parameters
    n_features = 5
    feature_dim = 10
    input_dim = (n_features + 1) * feature_dim  # +1 for bias feature
    
    # Generate data
    features, labels = generate_biased_data(
        n_samples=1000,
        n_features=n_features,
        feature_dim=feature_dim,
        bias_strength=0.8
    )
    
    # Define feature groups
    feature_groups = {
        'core_features': slice(0, n_features * feature_dim),
        'bias_feature': slice(n_features * feature_dim, input_dim)
    }
    
    # Create and train model
    model = SimpleTransformer(input_dim).to(device)
    print("Training initial model...")
    model = train_model(model, features, labels)
    
    # Evaluate initial model
    initial_acc = evaluate_model(model, features, labels)
    initial_gen = test_generalization(model, features, labels)
    print(f"\nInitial Model:")
    print(f"Training Accuracy: {initial_acc:.3f}")
    print(f"Generalization Test: {initial_gen:.3f}")
    
    # Detect spurious correlations
    detector = EnhancedDetector(model)
    scores = detector.detect_spurious_correlations(features, labels, feature_groups)
    
    print("\nSpurious Correlation Scores:")
    for group, metrics in scores.items():
        print(f"\n{group}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.3f}")
    
    # Apply mitigation
    mitigator = EnhancedMitigator(model, detector)
    print("\nApplying mitigation...")
    mitigator.mitigate_spurious_correlations(features, labels, feature_groups)
    
    # Evaluate after mitigation
    final_acc = evaluate_model(model, features, labels)
    final_gen = test_generalization(model, features, labels)
    print(f"\nAfter Mitigation:")
    print(f"Training Accuracy: {final_acc:.3f}")
    print(f"Generalization Test: {final_gen:.3f}")
    
    # Visualize results
    plt.figure(figsize=(10, 5))
    plt.bar(['Initial', 'Mitigated'], [initial_gen, final_gen])
    plt.title('Generalization Performance Before and After Mitigation')
    plt.ylabel('Accuracy')
    plt.show()