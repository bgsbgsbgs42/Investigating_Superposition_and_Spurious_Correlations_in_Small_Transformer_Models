# healthcare_analysis.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any

class HealthcareDataProcessor:
    def __init__(self, path: str):
        self.data = pd.read_csv(path)
        self.label_encoders = {}
        
    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clinical features
        clinical_features = [
            'Blood Pressure', 'BMI', 'Blood Glucose', 'Heart Rate',
            'Physical Activity'
        ]
        
        # Administrative/demographic features
        admin_features = [
            'Gender', 'Age', 'Occupation', 'Sleep Duration',
            'Quality of Sleep'
        ]
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Occupation']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        
        # Scale numerical features
        scaler = StandardScaler()
        self.data[clinical_features + admin_features] = scaler.fit_transform(
            self.data[clinical_features + admin_features]
        )
        
        # Create outcome variable (1 if stress level > median)
        median_stress = self.data['Stress Level'].median()
        y = (self.data['Stress Level'] > median_stress).astype(int)
        
        # Prepare feature matrix
        X = self.data[clinical_features + admin_features].values
        
        return torch.FloatTensor(X), torch.FloatTensor(y)

class HealthcareTransformer(nn.Module):
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
        x = self.transformer(x.unsqueeze(1))
        return torch.sigmoid(self.output(x.squeeze(1)))

class HealthcareDetailedAnalyzer:
    def __init__(self, model: nn.Module, features: torch.Tensor, labels: torch.Tensor):
        self.model = model
        self.features = features
        self.labels = labels
        
    def analyze_stress_patterns(self) -> Dict[str, float]:
        """Analyze patterns in stress level predictions"""
        clinical_indices = slice(0, 5)
        admin_indices = slice(5, 10)
        
        with torch.no_grad():
            base_pred = self.model(self.features.unsqueeze(1))
            
            # Clinical-only prediction
            clinical_features = self.features.clone()
            clinical_features[:, admin_indices] = 0
            clinical_pred = self.model(clinical_features.unsqueeze(1))
            
            # Admin-only prediction
            admin_features = self.features.clone()
            admin_features[:, clinical_indices] = 0
            admin_pred = self.model(admin_features.unsqueeze(1))
            
            return {
                'clinical_accuracy': ((clinical_pred > 0.5) == self.labels).float().mean().item(),
                'admin_accuracy': ((admin_pred > 0.5) == self.labels).float().mean().item(),
                'combined_accuracy': ((base_pred > 0.5) == self.labels).float().mean().item(),
                'clinical_contribution': torch.corrcoef(
                    torch.stack([base_pred.squeeze(), clinical_pred.squeeze()])
                )[0,1].item(),
                'admin_contribution': torch.corrcoef(
                    torch.stack([base_pred.squeeze(), admin_pred.squeeze()])
                )[0,1].item()
            }

    def analyze_interaction_effects(self) -> Dict[str, Any]:
        """Analyze interaction between clinical and administrative features"""
        interactions = np.zeros((5, 5))  # Clinical x Admin interactions
        
        for i in range(5):  # Clinical features
            for j in range(5):  # Admin features
                interactions[i, j] = self._measure_interaction(i, j+5)
                
        return {
            'interaction_matrix': interactions,
            'strongest_pairs': self._get_top_interactions(interactions),
            'interaction_strength': np.mean(np.abs(interactions))
        }
    
    def _measure_interaction(self, feat1: int, feat2: int) -> float:
        modified = self.features.clone()
        
        with torch.no_grad():
            base_pred = self.model(self.features.unsqueeze(1))
            
            modified[:, feat1] = 0
            effect1 = self.model(modified.unsqueeze(1))
            
            modified = self.features.clone()
            modified[:, feat2] = 0
            effect2 = self.model(modified.unsqueeze(1))
            
            modified[:, [feat1, feat2]] = 0
            joint_effect = self.model(modified.unsqueeze(1))
            
            interaction = torch.mean(
                torch.abs((base_pred - joint_effect) - 
                         ((base_pred - effect1) + (base_pred - effect2)))
            ).item()
            
        return interaction
    
    def _get_top_interactions(self, matrix: np.ndarray) -> list:
        indices = np.unravel_index(
            np.argsort(matrix.ravel())[-3:],
            matrix.shape
        )
        return list(zip(indices[0], indices[1]))
    
    def analyze_temporal_stability(self) -> Dict[str, Dict]:
        """Analyze prediction stability across different conditions"""
        with torch.no_grad():
            predictions = self.model(self.features.unsqueeze(1))
            
        stability = {}
        for i in range(self.features.shape[1]):
            feature_values = self.features[:, i]
            quartiles = torch.quantile(feature_values, torch.tensor([0.25, 0.5, 0.75]))
            
            pred_std = []
            for j in range(3):
                if j == 0:
                    mask = feature_values <= quartiles[0]
                elif j == 1:
                    mask = (feature_values > quartiles[0]) & (feature_values <= quartiles[1])
                else:
                    mask = feature_values > quartiles[1]
                    
                pred_std.append(torch.std(predictions[mask]).item())
                
            stability[f'feature_{i}'] = {
                'prediction_std': pred_std,
                'range_sensitivity': max(pred_std) - min(pred_std)
            }
            
        return stability

def analyze_subgroups(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, Dict]:
    """Analyze model performance across different subgroups"""
    subgroup_analysis = {}
    
    # Define subgroups based on age and occupation
    age_groups = {
        'young': features[:, 6] < -0.5,
        'middle': (features[:, 6] >= -0.5) & (features[:, 6] <= 0.5),
        'older': features[:, 6] > 0.5
    }
    
    occupations = torch.unique(features[:, 7])
    
    with torch.no_grad():
        predictions = model(features.unsqueeze(1))
        
        # Age group analysis
        for group_name, mask in age_groups.items():
            group_metrics = {
                'accuracy': ((predictions[mask] > 0.5) == labels[mask]).float().mean().item(),
                'bias': torch.mean(predictions[mask] - labels[mask]).item(),
                'feature_importance': analyze_feature_importance(
                    model, features[mask].unsqueeze(1), labels[mask]
                )
            }
            subgroup_analysis[f'age_{group_name}'] = group_metrics
        
        # Occupation analysis
        for occ in occupations:
            mask = features[:, 7] == occ
            occ_metrics = {
                'accuracy': ((predictions[mask] > 0.5) == labels[mask]).float().mean().item(),
                'bias': torch.mean(predictions[mask] - labels[mask]).item(),
                'feature_importance': analyze_feature_importance(
                    model, features[mask].unsqueeze(1), labels[mask]
                )
            }
            subgroup_analysis[f'occupation_{int(occ)}'] = occ_metrics
    
    return subgroup_analysis

def analyze_feature_importance(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> list:
    """Analyze importance of each feature"""
    with torch.no_grad():
        base_pred = model(features)
        importance = []
        
        for i in range(features.shape[2]):
            modified = features.clone()
            modified[:, :, i] = 0
            importance.append(
                torch.abs(model(modified) - base_pred).mean().item()
            )
        
        return importance

def train_healthcare_model(path: str) -> Dict[str, Any]:
    """Train healthcare model and return analysis results"""
    # Process data
    processor = HealthcareDataProcessor(path)
    features, labels = processor.preprocess_data()
    
    # Initialize model
    model = HealthcareTransformer(input_dim=features.shape[1])
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(features.unsqueeze(1))
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Analyze results
    analyzer = HealthcareDetailedAnalyzer(model, features, labels)
    stress_patterns = analyzer.analyze_stress_patterns()
    interactions = analyzer.analyze_interaction_effects()
    stability = analyzer.analyze_temporal_stability()
    subgroups = analyze_subgroups(model, features.unsqueeze(1), labels)
    
    return {
        'model': model,
        'features': features,
        'labels': labels,
        'stress_patterns': stress_patterns,
        'interactions': interactions,
        'stability': stability,
        'subgroups': subgroups
    }

if __name__ == "__main__":
    # Example usage
    results = train_healthcare_model("healthcare_data.csv")
    
    print("\nStress Pattern Analysis:")
    for metric, value in results['stress_patterns'].items():
        print(f"{metric}: {value:.3f}")
    
    print("\nTop Feature Interactions:")
    for i, j in results['interactions']['strongest_pairs']:
        print(f"Clinical feature {i} x Admin feature {j}: {results['interactions']['interaction_matrix'][i,j]:.3f}")
    
    print("\nSubgroup Analysis:")
    for group, metrics in results['subgroups'].items():
        print(f"\n{group}:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Bias: {metrics['bias']:.3f}")