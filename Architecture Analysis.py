import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data Generation
def generate_synthetic_data(n_samples=1000, n_features=10, feature_dim=10):
    """Generate synthetic data for architecture testing"""
    # Generate features
    features = np.random.normal(0, 1, (n_samples, n_features * feature_dim))
    
    # Create labels based on non-linear combination of features
    labels = (np.sum(features[:, :feature_dim*3]**2, axis=1) > feature_dim*1.5).astype(float)
    
    # Add some noise to labels
    labels = np.where(np.random.random(n_samples) < 0.9, labels, 1 - labels)
    
    return torch.FloatTensor(features).to(device), torch.FloatTensor(labels).to(device)

# 2. Transformer Architecture Variants
class TransformerVariants:
    class ParallelTransformer(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, n_branches: int = 4):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            self.parallel_branches = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, 
                    nhead=4,
                    dim_feedforward=hidden_dim*4,
                    batch_first=True
                )
                for _ in range(n_branches)
            ])
            
            self.output = nn.Linear(hidden_dim * n_branches, 1)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x.unsqueeze(1))  # Add sequence dimension
            branch_outputs = [branch(x) for branch in self.parallel_branches]
            combined = torch.cat(branch_outputs, dim=-1)
            return torch.sigmoid(self.output(combined.mean(dim=1))).squeeze()

    class HierarchicalTransformer(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # Local processing
            self.local_transformer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            )
            
            # Global processing
            self.global_transformer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            )
            
            self.output = nn.Linear(hidden_dim, 1)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x.unsqueeze(1))  # Add sequence dimension
            
            # Local processing in windows
            batch_size, seq_len = x.shape[:2]
            window_size = seq_len // 4
            
            local_outputs = []
            for i in range(0, seq_len, window_size):
                window = x[:, i:i+window_size]
                if window.size(1) > 0:  # Handle all windows
                    local_outputs.append(self.local_transformer(window))
                    
            x = torch.cat(local_outputs, dim=1)
            
            # Global processing
            x = self.global_transformer(x)
            return torch.sigmoid(self.output(x.mean(dim=1))).squeeze()

    class GatedTransformer(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            self.content_transformer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            )
            
            self.gate_transformer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            )
            
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x.unsqueeze(1))  # Add sequence dimension
            
            content = self.content_transformer(x)
            gates = torch.sigmoid(self.gate_proj(self.gate_transformer(x)))
            
            gated_output = content * gates
            return torch.sigmoid(self.output(gated_output.mean(dim=1))).squeeze()

# 3. Architecture Analyzer
class ArchitectureAnalyzer:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def analyze_architecture(self, model: nn.Module) -> Dict:
        metrics = {}
        
        # Analyze representation structure
        repr_metrics = self._analyze_representations(model)
        metrics['representation'] = repr_metrics
        
        # Analyze feature attribution
        attribution = self._analyze_feature_attribution(model)
        metrics['attribution'] = attribution
        
        # Analyze robustness
        robustness = self._analyze_robustness(model)
        metrics['robustness'] = robustness
        
        return metrics
    
    def _analyze_representations(self, model: nn.Module) -> Dict:
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        with torch.no_grad():
            _ = model(self.features)
        
        metrics = {}
        for name, acts in activations.items():
            # Calculate representation metrics
            acts_flat = acts.reshape(-1, acts.shape[-1])
            
            # SVD analysis
            U, S, V = torch.svd(acts_flat)
            
            metrics[name] = {
                'rank': torch.sum(S > 0.01 * S[0]).item(),
                'condition_number': (S[0] / S[-1]).item(),
                'sparsity': torch.mean((acts_flat.abs() < 0.01).float()).item()
            }
            
        for hook in hooks:
            hook.remove()
            
        return metrics
    
    def _analyze_feature_attribution(self, model: nn.Module) -> Dict:
        attributions = {}
        
        # Simple gradient-based attribution
        self.features.requires_grad_(True)
        outputs = model(self.features)
        grads = torch.autograd.grad(outputs.sum(), self.features)[0]
        
        for i in range(self.features.shape[-1]):
            attributions[f'feature_{i}'] = grads[:, :, i].abs().mean().item()
            
        self.features.requires_grad_(False)
        return attributions
    
    def _analyze_robustness(self, model: nn.Module) -> Dict:
        metrics = {}
        
        # Noise robustness
        noise_levels = [0.01, 0.05, 0.1]
        noise_impact = []
        
        with torch.no_grad():
            orig_pred = model(self.features)
            for noise in noise_levels:
                noisy_features = self.features + torch.randn_like(self.features) * noise
                noisy_pred = model(noisy_features)
                impact = torch.mean(torch.abs(orig_pred - noisy_pred)).item()
                noise_impact.append(impact)
                
        metrics['noise_sensitivity'] = np.mean(noise_impact)
        
        # Feature ablation
        ablation_impact = []
        with torch.no_grad():
            orig_pred = model(self.features)
            for i in range(self.features.shape[-1]):
                ablated = self.features.clone()
                ablated[:, :, i] = 0
                ablated_pred = model(ablated)
                impact = torch.mean(torch.abs(orig_pred - ablated_pred)).item()
                ablation_impact.append(impact)
                
        metrics['feature_sensitivity'] = np.mean(ablation_impact)
        
        return metrics

# 4. Comparison Function
def compare_architectures(features, labels):
    variants = TransformerVariants()
    input_dim = features.shape[-1]
    
    architectures = {
        'parallel': variants.ParallelTransformer(input_dim).to(device),
        'hierarchical': variants.HierarchicalTransformer(input_dim).to(device),
        'gated': variants.GatedTransformer(input_dim).to(device)
    }
    
    analyzer = ArchitectureAnalyzer(features, labels)
    results = {}
    
    for name, model in architectures.items():
        print(f"\nAnalyzing {name} architecture...")
        metrics = analyzer.analyze_architecture(model)
        results[name] = metrics
        
    return results

# 5. Visualization Function
def visualize_results(results):
    # Prepare data for plotting
    architectures = list(results.keys())
    
    # Representation metrics
    avg_ranks = [np.mean([m['rank'] for m in res['representation'].values()]) 
                for res in results.values()]
    condition_numbers = [np.mean([m['condition_number'] for m in res['representation'].values()]) 
                        for res in results.values()]
    
    # Robustness metrics
    noise_sensitivity = [res['robustness']['noise_sensitivity'] 
                        for res in results.values()]
    feature_sensitivity = [res['robustness']['feature_sensitivity'] 
                          for res in results.values()]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Average Rank
    axes[0,0].bar(architectures, avg_ranks)
    axes[0,0].set_title('Representation Rank')
    axes[0,0].set_ylabel('Average Rank')
    
    # Plot 2: Condition Number
    axes[0,1].bar(architectures, condition_numbers)
    axes[0,1].set_title('Representation Condition Number')
    axes[0,1].set_ylabel('Condition Number')
    
    # Plot 3: Noise Sensitivity
    axes[1,0].bar(architectures, noise_sensitivity)
    axes[1,0].set_title('Noise Sensitivity')
    axes[1,0].set_ylabel('Prediction Change')
    
    # Plot 4: Feature Sensitivity
    axes[1,1].bar(architectures, feature_sensitivity)
    axes[1,1].set_title('Feature Sensitivity')
    axes[1,1].set_ylabel('Prediction Change')
    
    plt.tight_layout()
    plt.show()

# 6. Main Execution
if __name__ == "__main__":
    # Generate data
    features, labels = generate_synthetic_data()
    
    # Compare architectures
    results = compare_architectures(features, labels)
    
    # Print results
    print("\nArchitecture Comparison Results:")
    for arch, metrics in results.items():
        print(f"\n{arch.upper()}:")
        print(f"Average Rank: {np.mean([m['rank'] for m in metrics['representation'].values()]):.2f}")
        print(f"Feature Attribution Variance: {np.var(list(metrics['attribution'].values())):.3f}")
        print(f"Noise Sensitivity: {metrics['robustness']['noise_sensitivity']:.3f}")
        print(f"Feature Sensitivity: {metrics['robustness']['feature_sensitivity']:.3f}")
    
    # Visualize results
    visualize_results(results)