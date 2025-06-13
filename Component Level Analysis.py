import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data Generation
def generate_synthetic_data(n_samples=1000, n_features=10, feature_dim=10):
    """Generate synthetic data for component testing"""
    # Generate features
    features = np.random.normal(0, 1, (n_samples, n_features * feature_dim))
    
    # Create labels based on non-linear combination of features
    labels = (np.sum(features[:, :feature_dim*3]**2, axis=1) > feature_dim*1.5).astype(float)
    
    return torch.FloatTensor(features).to(device), torch.FloatTensor(labels).to(device)

# 2. Transformer Architecture Variants (from previous architecture-test.py)
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

# 3. Component Analyzer
class ComponentAnalyzer:
    def __init__(self, model):
        self.model = model
        self.components = self._identify_components()
        
    def _identify_components(self):
        components = {
            'attention': [],
            'feedforward': [],
            'gating': [],
            'normalization': []
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                components['attention'].append((name, module))
            elif isinstance(module, nn.Linear) and not any(x in name for x in ['gate', 'output']):
                components['feedforward'].append((name, module))
            elif 'gate' in name and isinstance(module, nn.Linear):
                components['gating'].append((name, module))
            elif isinstance(module, nn.LayerNorm):
                components['normalization'].append((name, module))
                
        return components
    
    def analyze_attention_components(self, features):
        """Analyze attention mechanisms"""
        metrics = {}
        for name, module in self.components['attention']:
            with torch.no_grad():
                # Get attention patterns
                _, attn_weights = module(features, features, features)
                
                # Analyze attention focus
                focus = self._analyze_attention_focus(attn_weights)
                
                # Analyze head specialization
                specialization = self._analyze_head_specialization(attn_weights)
                
                metrics[name] = {
                    'focus': focus,
                    'specialization': specialization
                }
        return metrics
    
    def _analyze_attention_focus(self, attention_weights):
        # Calculate attention entropy and sparsity
        probs = torch.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        sparsity = torch.mean((probs < 0.01).float())
        
        return {
            'entropy': entropy.mean().item(),
            'sparsity': sparsity.item()
        }
    
    def _analyze_head_specialization(self, attention_weights):
        # Calculate head diversity
        head_patterns = attention_weights.mean(dim=1)
        similarity = torch.corrcoef(head_patterns.reshape(head_patterns.shape[0], -1))
        diversity = torch.mean(torch.abs(similarity - torch.eye(similarity.shape[0], device=device))).item()
        
        return {
            'head_diversity': diversity
        }
    
    def analyze_feedforward_components(self, features):
        """Analyze feedforward networks"""
        metrics = {}
        for name, module in self.components['feedforward']:
            with torch.no_grad():
                # Analyze weight distribution
                weight_stats = self._analyze_weight_distribution(module)
                
                # Analyze activation patterns
                act_patterns = self._analyze_activation_patterns(module, features)
                
                metrics[name] = {
                    'weight_stats': weight_stats,
                    'activation_patterns': act_patterns
                }
        return metrics
    
    def _analyze_weight_distribution(self, module):
        weights = module.weight.data
        return {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'sparsity': torch.mean((weights.abs() < 0.01).float()).item()
        }
    
    def _analyze_activation_patterns(self, module, features):
        output = module(features)
        return {
            'activation_mean': output.mean().item(),
            'activation_std': output.std().item(),
            'dead_neurons': torch.mean((output.abs().mean(dim=0) < 0.01).float()).item()
        }
    
    def analyze_component_interactions(self, features):
        """Analyze interactions between components"""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for component_type in self.components:
            for name, module in self.components[component_type]:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        with torch.no_grad():
            _ = self.model(features)
        
        # Calculate interaction metrics
        interactions = {}
        for name1, acts1 in activations.items():
            for name2, acts2 in activations.items():
                if name1 < name2:
                    # Flatten activations for correlation calculation
                    flat1 = acts1.reshape(-1, acts1.shape[-1]).T
                    flat2 = acts2.reshape(-1, acts2.shape[-1]).T
                    
                    # Calculate correlation matrix
                    corr_matrix = torch.corrcoef(torch.cat([flat1, flat2], dim=0))
                    n = flat1.shape[0]
                    cross_corr = corr_matrix[:n, n:].abs().mean()
                    
                    interactions[f"{name1}_x_{name2}"] = {
                        'correlation': cross_corr.item()
                    }
                    
        for hook in hooks:
            hook.remove()
            
        return interactions

# 4. Comparison Function
def compare_component_behaviors(architectures, features):
    results = {}
    for name, model in architectures.items():
        analyzer = ComponentAnalyzer(model)
        
        results[name] = {
            'attention': analyzer.analyze_attention_components(features),
            'feedforward': analyzer.analyze_feedforward_components(features),
            'interactions': analyzer.analyze_component_interactions(features)
        }
        
    return results

# 5. Visualization Function
def visualize_component_analysis(results):
    # Prepare data for plotting
    architectures = list(results.keys())
    
    # Attention metrics
    attention_entropy = [np.mean([m['focus']['entropy'] for m in res['attention'].values()]) 
                        for res in results.values()]
    head_diversity = [np.mean([m['specialization']['head_diversity'] for m in res['attention'].values()]) 
                     for res in results.values()]
    
    # Feedforward metrics
    ffn_sparsity = [np.mean([m['weight_stats']['sparsity'] for m in res['feedforward'].values()]) 
                   for res in results.values()]
    
    # Interaction metrics
    component_corr = [np.mean([m['correlation'] for m in res['interactions'].values()]) 
                     for res in results.values()]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Attention Entropy
    axes[0,0].bar(architectures, attention_entropy)
    axes[0,0].set_title('Attention Entropy')
    axes[0,0].set_ylabel('Entropy')
    
    # Plot 2: Head Diversity
    axes[0,1].bar(architectures, head_diversity)
    axes[0,1].set_title('Attention Head Diversity')
    axes[0,1].set_ylabel('Diversity Score')
    
    # Plot 3: FFN Sparsity
    axes[1,0].bar(architectures, ffn_sparsity)
    axes[1,0].set_title('Feedforward Sparsity')
    axes[1,0].set_ylabel('Sparsity')
    
    # Plot 4: Component Correlation
    axes[1,1].bar(architectures, component_corr)
    axes[1,1].set_title('Component Interaction')
    axes[1,1].set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.show()

# 6. Main Execution
if __name__ == "__main__":
    # Generate data
    features, labels = generate_synthetic_data()
    
    # Create architectures
    variants = TransformerVariants()
    input_dim = features.shape[-1]
    
    architectures = {
        'parallel': variants.ParallelTransformer(input_dim).to(device),
        'hierarchical': variants.HierarchicalTransformer(input_dim).to(device),
        'gated': variants.GatedTransformer(input_dim).to(device)
    }
    
    # Analyze components
    results = compare_component_behaviors(architectures, features)
    
    # Print summary
    print("\nComponent Analysis Summary:")
    for arch, metrics in results.items():
        print(f"\n{arch.upper()}:")
        print(f"Attention Entropy: {np.mean([m['focus']['entropy'] for m in metrics['attention'].values()]):.3f}")
        print(f"Head Diversity: {np.mean([m['specialization']['head_diversity'] for m in metrics['attention'].values()]):.3f}")
        print(f"FFN Sparsity: {np.mean([m['weight_stats']['sparsity'] for m in metrics['feedforward'].values()]):.3f}")
        print(f"Component Correlation: {np.mean([m['correlation'] for m in metrics['interactions'].values()]):.3f}")
    
    # Visualize results
    visualize_component_analysis(results)