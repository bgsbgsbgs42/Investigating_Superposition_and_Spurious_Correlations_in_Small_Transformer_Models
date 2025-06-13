# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

def plot_healthcare_results(results: Dict[str, Any]) -> plt.Figure:
    """Visualize healthcare analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Stress patterns
    stress_data = pd.DataFrame.from_dict(results['stress_patterns'], orient='index', columns=['Value'])
    stress_data.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Stress Pattern Analysis')
    
    # Feature interactions
    sns.heatmap(results['interactions']['interaction_matrix'], ax=axes[0,1])
    axes[0,1].set_title('Feature Interaction Matrix')
    
    # Stability analysis
    stability_data = pd.DataFrame({
        'range_sensitivity': [v['range_sensitivity'] for v in results['stability'].values()]
    }, index=results['stability'].keys())
    stability_data.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Feature Stability Analysis')
    
    # Subgroup accuracy
    subgroup_data = pd.DataFrame({
        'accuracy': [v['accuracy'] for v in results['subgroups'].values()]
    }, index=results['subgroups'].keys())
    subgroup_data.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Subgroup Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_btc_results(results: Dict[str, Any]) -> plt.Figure:
    """Visualize BTC analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Volatility regimes
    regime_data = pd.DataFrame({
        'accuracy': [v['accuracy'] for v in results['volatility_analysis'].values()],
        'confidence': [v['confidence'] for v in results['volatility_analysis'].values()]
    }, index=results['volatility_analysis'].keys())
    regime_data.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Volatility Regime Performance')
    
    # Cycle correlations
    cycle_data = pd.DataFrame.from_dict(results['cycle_analysis']['cycle_correlations'], orient='index', columns=['Correlation'])
    cycle_data.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Market Cycle Correlations')
    
    # Feature importance
    feature_data = pd.DataFrame.from_dict(results['cycle_analysis']['feature_importance'], orient='index', columns=['Importance'])
    feature_data.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Feature Group Importance')
    
    # Feature importance by regime
    importance_data = []
    for regime, metrics in results['volatility_analysis'].items():
        for i, imp in enumerate(metrics['feature_importance']):
            importance_data.append({'regime': regime, 'feature': f'Feature {i}', 'importance': imp})
    importance_df = pd.DataFrame(importance_data)
    sns.barplot(data=importance_df, x='feature', y='importance', hue='regime', ax=axes[1,1])
    axes[1,1].set_title('Feature Importance by Regime')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage (would need actual results from analyses)
    healthcare_results = {}  # Replace with actual results
    btc_results = {}  # Replace with actual results
    
    healthcare_fig = plot_healthcare_results(healthcare_results)
    btc_fig = plot_btc_results(btc_results)
    
    plt.show()