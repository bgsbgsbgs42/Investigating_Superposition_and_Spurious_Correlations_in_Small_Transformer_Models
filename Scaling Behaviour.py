import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data Generation
def generate_synthetic_data(n_samples=1000, n_features=10, feature_dim=10):
    """Generate synthetic data for scaling analysis"""
    # Generate features
    features = np.random.normal(0, 1, (n_samples, n_features * feature_dim))
    
    # Create labels based on non-linear combination of features
    labels = (np.sum(features[:, :feature_dim*3]**2, axis=1) > feature_dim*1.5).astype(float)
    
    return torch.FloatTensor(features).to(device), torch.FloatTensor(labels).to(device)

# 2. Large Transformer Model
class LargeTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 hidden_dim: int = 768):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multiple attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers * 2)  # One for each attention and FF layer
        ])
        
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for i in range(len(self.attention_layers)):
            # Attention block
            attn_out, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms[i*2](x + attn_out)
            
            # Feed-forward block
            ff_out = self.ff_layers[i](x)
            x = self.layer_norms[i*2+1](x + ff_out)
            
        return torch.sigmoid(self.output(x.mean(dim=1))).squeeze()

# 3. Scaling Analyzer
class ScalingAnalyzer:
    def __init__(self):
        pass
        
    def analyze_representation_scaling(self, 
                                    features: torch.Tensor,
                                    hidden_dims: List[int] = [256, 512, 768, 1024]) -> Dict:
        """Analyze how representations scale with model width"""
        scaling_metrics = {}
        
        for dim in hidden_dims:
            print(f"Analyzing hidden_dim={dim}...")
            model = LargeTransformer(input_dim=features.shape[-1], hidden_dim=dim).to(device)
            
            # Quick training
            self._quick_train(model, features)
            
            # Analyze capacity
            capacity_metrics = self._analyze_capacity(model, features)
            
            # Analyze feature interactions
            interaction_metrics = self._analyze_feature_interactions(model, features)
            
            scaling_metrics[dim] = {
                'capacity': capacity_metrics,
                'interactions': interaction_metrics
            }
                
        return scaling_metrics
    
    def analyze_depth_scaling(self,
                            features: torch.Tensor,
                            n_layers_list: List[int] = [2, 4, 8, 12, 16]) -> Dict:
        """Analyze how model behavior changes with depth"""
        depth_metrics = {}
        
        for n_layers in n_layers_list:
            print(f"Analyzing n_layers={n_layers}...")
            model = LargeTransformer(
                input_dim=features.shape[-1],
                n_layers=n_layers
            ).to(device)
            
            # Quick training
            self._quick_train(model, features)
            
            # Analyze gradient flow
            grad_metrics = self._analyze_gradient_flow(model, features)
            
            # Analyze layer specialization
            specialization = self._analyze_layer_specialization(model, features)
            
            depth_metrics[n_layers] = {
                'gradient_metrics': grad_metrics,
                'specialization': specialization
            }
            
        return depth_metrics
    
    def _quick_train(self, model, features, epochs=5):
        """Quick training for analysis purposes"""
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCELoss()
        
        # Dummy labels for analysis (not used for actual training)
        dummy_labels = torch.randint(0, 2, (len(features),)).float().to(device)
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, dummy_labels)
            loss.backward()
            optimizer.step()
    
    def _analyze_capacity(self, model, features):
        """Analyze model capacity utilization"""
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        with torch.no_grad():
            _ = model(features)
        
        capacity_metrics = {}
        for name, acts in activations.items():
            # Flatten activations
            flat_acts = acts.reshape(-1, acts.shape[-1])
            
            # Measure activation sparsity
            sparsity = torch.mean((flat_acts.abs() < 0.01).float()).item()
            
            # Measure activation range
            dynamic_range = (flat_acts.max() - flat_acts.min()).item()
            
            # Measure activation entropy
            act_hist = torch.histc(flat_acts.float(), bins=50)
            act_probs = act_hist / act_hist.sum()
            entropy = -torch.sum(act_probs * torch.log2(act_probs + 1e-10)).item()
            
            capacity_metrics[name] = {
                'sparsity': sparsity,
                'dynamic_range': dynamic_range,
                'entropy': entropy
            }
            
        for hook in hooks:
            hook.remove()
            
        return capacity_metrics
    
    def _analyze_feature_interactions(self, model, features):
        """Analyze how feature interactions scale"""
        feature_dim = features.shape[-1]
        interaction_strengths = torch.zeros(feature_dim, feature_dim, device=device)
        
        with torch.no_grad():
            base_output = model(features)
            
            for i in range(feature_dim):
                for j in range(i+1, feature_dim):
                    # Zero out feature i
                    mod_features = features.clone()
                    mod_features[..., i] = 0
                    output_i = model(mod_features)
                    
                    # Zero out feature j
                    mod_features = features.clone()
                    mod_features[..., j] = 0
                    output_j = model(mod_features)
                    
                    # Zero out both
                    mod_features = features.clone()
                    mod_features[..., [i,j]] = 0
                    output_ij = model(mod_features)
                    
                    # Calculate interaction strength
                    interaction = torch.abs(
                        (base_output - output_ij) - 
                        ((base_output - output_i) + (base_output - output_j))
                    ).mean()
                    
                    interaction_strengths[i,j] = interaction
                    interaction_strengths[j,i] = interaction
                
        return {
            'mean_interaction': interaction_strengths.mean().item(),
            'max_interaction': interaction_strengths.max().item()
        }
    
    def _analyze_gradient_flow(self, model, features):
        """Analyze gradient flow through layers"""
        gradients = []
        
        def grad_hook(name):
            def hook(grad):
                gradients.append((name, grad.detach()))
            return hook
            
        handles = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                handle = param.register_hook(grad_hook(name))
                handles.append(handle)
                
        # Forward and backward pass
        output = model(features)
        output.mean().backward()
        
        # Calculate metrics
        grad_metrics = {}
        for name, grad in gradients:
            grad_metrics[name] = {
                'magnitude': grad.norm().item(),
                'variance': grad.var().item()
            }
            
        for handle in handles:
            handle.remove()
            
        return grad_metrics
    
    def _analyze_layer_specialization(self, model, features):
        """Analyze how layers specialize"""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.MultiheadAttention, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        with torch.no_grad():
            _ = model(features)
        
        specialization = {}
        for name, acts in activations.items():
            if isinstance(acts, tuple):
                acts = acts[0]  # For attention layers
                
            # Flatten activations
            flat_acts = acts.reshape(-1, acts.shape[-1])
            
            # Calculate feature selectivity
            mean_acts = torch.mean(flat_acts, dim=0)
            selectivity = torch.std(mean_acts).item()
            
            # Calculate activation patterns
            patterns = torch.corrcoef(flat_acts.T)
            pattern_diversity = torch.mean(torch.abs(patterns - torch.eye(patterns.shape[0], device=device))).item()
            
            specialization[name] = {
                'selectivity': selectivity,
                'pattern_diversity': pattern_diversity
            }
            
        for hook in hooks:
            hook.remove()
            
        return specialization

# 4. Visualization Functions
def plot_scaling_results(width_results, depth_results):
    """Visualize scaling trends"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Width scaling plots
    widths = sorted(width_results.keys())
    
    # Capacity utilization
    capacity = [np.mean([m['entropy'] for m in width_results[w]['capacity'].values()]) for w in widths]
    axes[0,0].plot(widths, capacity)
    axes[0,0].set_title('Capacity Utilization vs Width')
    axes[0,0].set_xlabel('Hidden Dimension')
    axes[0,0].set_ylabel('Activation Entropy')
    
    # Feature interactions
    interactions = [width_results[w]['interactions']['mean_interaction'] for w in widths]
    axes[0,1].plot(widths, interactions)
    axes[0,1].set_title('Feature Interactions vs Width')
    axes[0,1].set_xlabel('Hidden Dimension')
    axes[0,1].set_ylabel('Interaction Strength')
    
    # Representation sparsity
    sparsity = [np.mean([m['sparsity'] for m in width_results[w]['capacity'].values()]) for w in widths]
    axes[0,2].plot(widths, sparsity)
    axes[0,2].set_title('Representation Sparsity vs Width')
    axes[0,2].set_xlabel('Hidden Dimension')
    axes[0,2].set_ylabel('Sparsity')
    
    # Depth scaling plots
    depths = sorted(depth_results.keys())
    
    # Gradient magnitude
    grad_mag = [np.mean([g['magnitude'] for g in depth_results[d]['gradient_metrics'].values()]) for d in depths]
    axes[1,0].plot(depths, grad_mag)
    axes[1,0].set_title('Gradient Magnitude vs Depth')
    axes[1,0].set_xlabel('Number of Layers')
    axes[1,0].set_ylabel('Gradient Norm')
    
    # Layer specialization
    specialization = [np.mean([s['selectivity'] for s in depth_results[d]['specialization'].values()]) for d in depths]
    axes[1,1].plot(depths, specialization)
    axes[1,1].set_title('Layer Specialization vs Depth')
    axes[1,1].set_xlabel('Number of Layers')
    axes[1,1].set_ylabel('Selectivity')
    
    # Pattern diversity
    diversity = [np.mean([s['pattern_diversity'] for s in depth_results[d]['specialization'].values()]) for d in depths]
    axes[1,2].plot(depths, diversity)
    axes[1,2].set_title('Pattern Diversity vs Depth')
    axes[1,2].set_xlabel('Number of Layers')
    axes[1,2].set_ylabel('Diversity Score')
    
    plt.tight_layout()
    plt.show()

# 5. Main Execution
if __name__ == "__main__":
    # Generate data
    features, labels = generate_synthetic_data()
    
    # Initialize analyzer
    analyzer = ScalingAnalyzer()
    
    # Run width scaling analysis
    print("Running width scaling analysis...")
    width_results = analyzer.analyze_representation_scaling(features)
    
    # Run depth scaling analysis
    print("\nRunning depth scaling analysis...")
    depth_results = analyzer.analyze_depth_scaling(features)
    
    # Print summary trends
    def print_trends(width_results, depth_results):
        print("\nWidth Scaling Trends:")
        widths = sorted(width_results.keys())
        capacity = [np.mean([m['entropy'] for m in width_results[w]['capacity'].values()]) for w in widths]
        slope = np.polyfit(widths, capacity, 1)[0]
        print(f"Capacity Utilization: {slope:.3f} per dimension increase")
        
        interactions = [width_results[w]['interactions']['mean_interaction'] for w in widths]
        slope = np.polyfit(widths, interactions, 1)[0]
        print(f"Feature Interactions: {slope:.3f} per dimension increase")
        
        print("\nDepth Scaling Trends:")
        depths = sorted(depth_results.keys())
        grad_mag = [np.mean([g['magnitude'] for g in depth_results[d]['gradient_metrics'].values()]) for d in depths]
        slope = np.polyfit(depths, grad_mag, 1)[0]
        print(f"Gradient Magnitude: {slope:.3f} per layer increase")
        
        specialization = [np.mean([s['selectivity'] for s in depth_results[d]['specialization'].values()]) for d in depths]
        slope = np.polyfit(depths, specialization, 1)[0]
        print(f"Layer Specialization: {slope:.3f} per layer increase")
    
    print_trends(width_results, depth_results)
    
    # Visualize results
    plot_scaling_results(width_results, depth_results)