# Transformer Architecture Analysis and Spurious Correlation and Superposition Detection

## Abstract

This repository contains the implementation and analysis framework for investigating spurious correlations and superposition and architectural behaviours in transformer-based models. The project provides comprehensive tools for analysing model behaviour across different architectural variants, detecting spurious correlations, and examining scaling behaviours in transformer networks.

## Project Overview

The research framework consists of several interconnected modules designed to facilitate rigorous analysis of transformer architectures:

- **Dataset Generation**: Synthetic data generators for financial and healthcare domains with controllable spurious correlations
- **Spurious Correlation and Superposition Detection**: Advanced detection algorithms with enhanced metrics for identifying problematic feature dependencies
- **Architectural Analysis**: Comparative analysis tools for different transformer variants including parallel, hierarchical, and gated architectures
- **Scaling Behaviour**: Investigation of representation scaling with model width and depth
- **Real-world Applications**: Analysis pipelines for Bitcoin price prediction and healthcare outcome prediction

## Repository Structure

```
├── Datasets.py                              # Synthetic data generation
├── Mitigation Strategies.py                 # Spurious correlation detection and mitigation
├── Scaling Behaviour.py                     # Model scaling analysis
├── Superposition + Spurious Correlation.py  # Superposition analysis framework
├── Architecture Analysis.py                 # Transformer variant comparison
├── Component Level Analysis.py              # Individual component behaviour analysis
├── healthcare.py                           # Healthcare domain analysis
├── bitcoin.py                              # Financial domain analysis
└── visualisation.py                       # Result visualisation utilities
```

## Requirements

### Dependencies

The project requires Python 3.8+ and the following packages:

```bash
pip install torch>=1.9.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0
pip install scipy>=1.7.0
pip install yfinance>=0.1.70
```

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support and 16GB+ RAM for larger-scale experiments
- The framework automatically detects and utilises available CUDA devices

## Getting Started

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd transformer-analysis
pip install -r requirements.txt
```

### 2. Basic Usage

#### Generate Synthetic Datasets

Begin by creating synthetic datasets with controllable spurious correlations:

```python
from Datasets import generate_financial_data, generate_healthcare_data

# Generate financial dataset with realistic market patterns
financial_data = generate_financial_data(n_samples=1000, n_days=252)

# Generate healthcare dataset with patient trajectories
healthcare_data = generate_healthcare_data(n_patients=1000, n_visits=10)
```

#### Spurious Correlation and Superposition Analysis

Detect and mitigate spurious correlations and superposition in your models:

```python
from Mitigation_Strategies import EnhancedDetector, EnhancedMitigator

# Initialise detection framework
detector = EnhancedDetector(model, threshold=0.7)

# Define feature groups for analysis
feature_groups = {
    'core_features': slice(0, 50),
    'spurious_features': slice(50, 60)
}

# Detect spurious correlations
scores = detector.detect_spurious_correlations(features, labels, feature_groups)

# Apply mitigation strategies
mitigator = EnhancedMitigator(model, detector)
mitigator.mitigate_spurious_correlations(features, labels, feature_groups)
```

#### Architectural Comparison

Compare different transformer variants:

```python
from Architecture_Analysis import TransformerVariants, ArchitectureAnalyzer

# Initialise transformer variants
variants = TransformerVariants()
models = {
    'parallel': variants.ParallelTransformer(input_dim),
    'hierarchical': variants.HierarchicalTransformer(input_dim),
    'gated': variants.GatedTransformer(input_dim)
}

# Analyse architectural differences
analyzer = ArchitectureAnalyzer(features, labels)
results = {name: analyzer.analyze_architecture(model) 
          for name, model in models.items()}
```

### 3. Advanced Analysis

#### Scaling Behaviour Investigation

Examine how model behaviour changes with scale:

```python
from Scaling_Behaviour import ScalingAnalyzer

analyzer = ScalingAnalyzer()

# Analyse width scaling
width_results = analyzer.analyze_representation_scaling(
    features, hidden_dims=[256, 512, 768, 1024]
)

# Analyse depth scaling
depth_results = analyzer.analyze_depth_scaling(
    features, n_layers_list=[2, 4, 8, 12, 16]
)
```

#### Superposition Analysis

Investigate feature superposition patterns:

```python
from Superposition_Spurious_Correlation import SuperpositionAnalyzer

analyzer = SuperpositionAnalyzer(model)

# Visualise activation space
analyzer.visualize_superposition(features, 'input_proj')

# Analyse feature overlap
overlap_matrix = analyzer.analyze_feature_overlap(
    features, 'input_proj', feature_dims
)
```

### 4. Real-world Applications

#### Healthcare Analysis

Apply the framework to healthcare outcome prediction:

```python
from healthcare import train_healthcare_model

# Train and analyse healthcare model
results = train_healthcare_model("path/to/healthcare_data.csv")

# Examine stress pattern analysis
print("Stress Pattern Analysis:", results['stress_patterns'])
print("Feature Interactions:", results['interactions']['strongest_pairs'])
```

#### Bitcoin Price Analysis

Investigate cryptocurrency prediction models:

```python
from bitcoin import train_btc_model

# Train and analyse Bitcoin prediction model
results = train_btc_model()

# Examine volatility regime performance
print("Volatility Analysis:", results['volatility_analysis'])
print("Market Cycle Dependencies:", results['cycle_analysis'])
```

## Experimental Workflows

### Comprehensive Model Analysis Pipeline

For a complete analysis of a transformer model, follow this workflow:

```python
# 1. Data Generation
features, labels = generate_synthetic_data_with_bias()

# 2. Model Training
model = train_transformer_model(features, labels)

# 3. Spurious Correlation Detection
spurious_scores = detect_spurious_correlations(model, features, labels)

# 4. Architectural Analysis
arch_metrics = analyze_architecture_properties(model, features)

# 5. Scaling Investigation
scaling_results = investigate_scaling_behaviour(features)

# 6. Visualisation
plot_comprehensive_results(spurious_scores, arch_metrics, scaling_results)
```

### Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

## Key Research Findings

The framework has been designed to facilitate investigation of several key research questions:

1. **Spurious Correlation Detection**: How do different detection metrics compare in identifying problematic feature dependencies?
2. **Architectural Differences**: What are the trade-offs between parallel, hierarchical, and gated transformer architectures?
3. **Scaling Laws**: How do representation properties change with model width and depth?
4. **Domain Generalisation**: How do spurious correlations manifest differently across healthcare and financial domains?

## Visualisation and Results

The framework provides comprehensive visualisation tools:

```python
from visualisation import plot_healthcare_results, plot_btc_results

# Generate publication-ready figures
healthcare_fig = plot_healthcare_results(healthcare_results)
btc_fig = plot_btc_results(btc_results)
```

## Configuration

### Model Hyperparameters

Default configurations are provided for all models but can be customised:

```python
model_config = {
    'hidden_dim': 64,
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1
}
```

### Detection Thresholds

Spurious correlation detection thresholds can be adjusted based on domain requirements:

```python
detection_config = {
    'threshold': 0.7,
    'min_correlation': 0.3,
    'significance_level': 0.05
}
```

## Contributing

This research framework is designed for academic use. When contributing:

1. Maintain reproducibility through fixed random seeds
2. Document all methodological choices
3. Provide comprehensive test cases for new detection methods
4. Follow established coding standards for research software

## Support

For questions regarding the implementation or methodological choices, please refer to the inline documentation or raise an issue in the repository.

---

**Note**: This framework is intended for research purposes. When applying to real-world applications, ensure appropriate validation and consider domain-specific constraints.
