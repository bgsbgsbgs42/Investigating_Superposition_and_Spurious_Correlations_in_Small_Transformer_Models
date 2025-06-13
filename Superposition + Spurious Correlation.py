import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data Generation
def generate_multifeature_data(n_samples=1000, n_features=5, feature_dim=10):
    """Generate synthetic data with multiple independent features"""
    features = np.zeros((n_samples, n_features * feature_dim))
    labels = np.zeros(n_samples)
    
    for i in range(n_features):
        start_idx = i * feature_dim
        end_idx = (i+1) * feature_dim
        feature_values = np.random.normal(i, 1, (n_samples, feature_dim))
        features[:, start_idx:end_idx] = feature_values
        labels += np.sum(feature_values, axis=1)  # Simple sum as label
        
    return features, labels

def generate_biased_data(n_samples=1000, n_features=5, feature_dim=10, bias_strength=0.8):
    """Generate dataset with intentional spurious correlations"""
    features, labels = generate_multifeature_data(n_samples, n_features, feature_dim)
    
    # Introduce spurious correlation
    bias_feature = np.random.randn(n_samples, feature_dim)
    bias_labels = (np.sum(bias_feature**2, axis=1) > feature_dim/2)
    
    # Mix true labels with bias
    mixed_labels = np.where(
        np.random.random(n_samples) < bias_strength,
        bias_labels,
        labels
    )
    
    # Concatenate bias feature
    biased_features = np.concatenate([features, bias_feature], axis=1)
    
    return biased_features, mixed_labels

# 2. Model Definition
class SmallTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=4, n_layers=2, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.input_proj(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
            
        x = x.mean(dim=1)  # Pool sequence
        return self.output_proj(x).squeeze(-1)

# 3. Training Function
def train_model(model, features, labels, n_epochs=100, batch_size=32):
    """Train model with early stopping"""
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    features = torch.FloatTensor(features).to(device)
    labels = torch.FloatTensor(labels).to(device)
    
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

# 4. Superposition Analysis
class SuperpositionAnalyzer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        
    def _setup_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(hook_fn(name))
    
    def collect_activations(self, features: torch.Tensor) -> dict:
        """Collect activations for input features"""
        self._setup_hooks()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(features)
        return self.activations.copy()
    
    def analyze_superposition(self, 
                            features: torch.Tensor, 
                            layer_name: str,
                            n_components: int = 3) -> Tuple[np.ndarray, PCA]:
        """Analyze superposition in specified layer using PCA"""
        activations = self.collect_activations(features)
        layer_activations = activations[layer_name].cpu().numpy()
        
        # Reshape if needed
        if len(layer_activations.shape) == 3:
            layer_activations = layer_activations.reshape(-1, layer_activations.shape[-1])
            
        # Perform PCA
        pca = PCA(n_components=n_components)
        projected_activations = pca.fit_transform(layer_activations)
        
        return projected_activations, pca
    
    def visualize_superposition(self,
                              features: torch.Tensor,
                              layer_name: str,
                              feature_labels: List[str] = None):
        """Create visualization of superposition patterns"""
        projected_acts, pca = self.analyze_superposition(features, layer_name)
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(projected_acts[:, 0],
                           projected_acts[:, 1],
                           projected_acts[:, 2],
                           c=range(len(projected_acts)),
                           cmap='viridis')
        
        # Add labels
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)')
        
        plt.title(f'Neuron Activation Space: {layer_name}')
        plt.colorbar(scatter, label='Sample Index')
        
        return fig
    
    def analyze_feature_overlap(self,
                              features: torch.Tensor,
                              layer_name: str,
                              feature_dims: List[int]) -> np.ndarray:
        """Analyze how different features overlap in neuron space"""
        _, pca = self.analyze_superposition(features, layer_name)
        
        # Get principal components for each feature dimension
        overlap_matrix = np.zeros((len(feature_dims), len(feature_dims)))
        start_idx = 0
        
        for i, dim1 in enumerate(feature_dims):
            for j, dim2 in enumerate(feature_dims):
                if i <= j:
                    # Calculate overlap using cosine similarity of PC loadings
                    pc1 = pca.components_[:, start_idx:start_idx + dim1]
                    pc2 = pca.components_[:, start_idx + dim1:start_idx + dim1 + dim2]
                    
                    similarity = np.abs(np.dot(pc1.flatten(), pc2.flatten())) / \
                               (np.linalg.norm(pc1) * np.linalg.norm(pc2))
                    
                    overlap_matrix[i, j] = similarity
                    overlap_matrix[j, i] = similarity
            
            start_idx += dim1
            
        return overlap_matrix
    
    def plot_feature_overlap(self,
                           features: torch.Tensor,
                           layer_name: str, 
                           feature_dims: List[int],
                           feature_names: List[str] = None):
        """Visualize feature overlap as a heatmap"""
        overlap_matrix = self.analyze_feature_overlap(features, layer_name, feature_dims)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(overlap_matrix, cmap='YlOrRd')
        plt.colorbar(label='Feature Overlap')
        
        if feature_names:
            plt.xticks(range(len(feature_names)), feature_names, rotation=45)
            plt.yticks(range(len(feature_names)), feature_names)
        
        plt.title(f'Feature Overlap Analysis: {layer_name}')
        plt.tight_layout()
        
        return plt.gcf()

# 5. Main Execution
if __name__ == "__main__":
    # Parameters
    n_features = 5
    feature_dim = 10
    input_dim = n_features * feature_dim
    
    # Generate data
    features, labels = generate_biased_data(
        n_samples=1000,
        n_features=n_features,
        feature_dim=feature_dim,
        bias_strength=0.8
    )
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).to(device)
    labels_tensor = torch.FloatTensor(labels).to(device)
    
    # Create and train model
    model = SmallTransformer(input_dim=input_dim).to(device)
    model = train_model(model, features_tensor, labels_tensor)
    
    # Feature metadata
    feature_dims = [feature_dim] * n_features
    feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Analyze superposition
    analyzer = SuperpositionAnalyzer(model)
    
    # Visualize activations
    activation_fig = analyzer.visualize_superposition(
        features_tensor[:100],  # Use first 100 samples for visualization
        'input_proj',
        feature_names
    )
    
    # Plot feature overlap
    overlap_fig = analyzer.plot_feature_overlap(
        features_tensor[:100],
        'input_proj',
        feature_dims,
        feature_names
    )
    
    # Show plots
    plt.show()