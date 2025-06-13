import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

def generate_financial_data(n_samples: int = 1000, n_days: int = 252) -> pd.DataFrame:
    """
    Generate synthetic financial dataset with realistic market patterns
    
    Args:
        n_samples: Number of samples to generate (not used directly, kept for consistency)
        n_days: Number of trading days to generate
        
    Returns:
        DataFrame containing synthetic financial data with:
        - Price data (Close)
        - Volume data
        - Technical indicators (SMA, RSI, Volatility)
        - Calendar effects (DayOfWeek, MonthEnd, QuarterEnd)
    """
    # Validate inputs
    if n_days < 30:
        raise ValueError("n_days must be at least 30 to calculate meaningful technical indicators")
    
    # Date range (business days only)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # Generate more realistic price data with volatility clustering
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0005, 0.02, n_days)
    
    # Add volatility clustering effect
    for i in range(1, n_days):
        if abs(returns[i-1]) > 0.03:  # If yesterday was volatile
            returns[i] *= 1.5  # Increase today's volatility
    
    # Generate prices from returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Calculate technical indicators
    def calculate_rsi(price_changes: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        gain = price_changes.where(price_changes > 0, 0).rolling(window).mean()
        loss = -price_changes.where(price_changes < 0, 0).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    price_changes = pd.Series(prices).pct_change()
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, n_days),
        # Technical features
        'SMA_20': pd.Series(prices).rolling(20).mean(),
        'RSI': calculate_rsi(price_changes),
        'Volatility': pd.Series(returns).rolling(20).std(),
        # Calendar effects (potentially spurious)
        'DayOfWeek': dates.dayofweek,
        'MonthEnd': dates.is_month_end.astype(int),
        'QuarterEnd': dates.is_quarter_end.astype(int),
        # Additional useful features
        'Daily_Return': returns,
        'Intraday_High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Intraday_Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    })
    
    # Clean any NaN values from rolling calculations
    data = data.dropna()
    
    return data

def generate_healthcare_data(n_patients: int = 1000, n_visits: int = 10) -> pd.DataFrame:
    """
    Generate synthetic healthcare dataset with realistic patient trajectories
    
    Args:
        n_patients: Number of unique patients to generate
        n_visits: Number of visits per patient
        
    Returns:
        DataFrame containing synthetic healthcare data with:
        - Clinical measurements (vitals, lab values)
        - Treatment information (medications)
        - Administrative data (insurance, facility)
        - Synthetic outcomes based on clinical factors
    """
    # Validate inputs
    if n_patients < 1:
        raise ValueError("Must generate at least 1 patient")
    if n_visits < 1:
        raise ValueError("Patients must have at least 1 visit")
    
    np.random.seed(42)  # For reproducibility
    data = []
    patient_ids = range(n_patients)
    
    # Generate some underlying health conditions that persist across visits
    health_conditions = {
        'hypertension': np.random.random(n_patients) < 0.2,
        'diabetes': np.random.random(n_patients) < 0.15,
        'copd': np.random.random(n_patients) < 0.1
    }
    
    for patient_id in patient_ids:
        # Generate baseline health metrics based on conditions
        base_hr = 75 + 5 * health_conditions['hypertension'][patient_id]
        base_bp = 120 + 10 * health_conditions['hypertension'][patient_id]
        base_temp = 37 + 0.3 * health_conditions['diabetes'][patient_id]
        
        # Generate progression factors that change slowly over visits
        hr_progression = np.random.normal(0, 0.1, n_visits).cumsum()
        bp_progression = np.random.normal(0, 0.15, n_visits).cumsum()
        
        for visit in range(n_visits):
            # Clinical features with realistic correlations
            heart_rate = max(40, min(150, 
                base_hr + hr_progression[visit] + np.random.normal(0, 3)))
            blood_pressure = max(80, min(200, 
                base_bp + bp_progression[visit] + np.random.normal(0, 5)))
            temperature = max(35, min(41, 
                base_temp + np.random.normal(0, 0.2)))
            respiratory_rate = max(8, min(30, 
                np.random.normal(16 + 2 * health_conditions['copd'][patient_id], 2)))
            oxygen_saturation = max(85, min(100, 
                np.random.normal(97 - 3 * health_conditions['copd'][patient_id], 1.5)))
            lab_values = np.random.normal(
                0.5 * health_conditions['diabetes'][patient_id], 0.8)
            medications = np.random.randint(0, 5)
            
            # Administrative features
            admission_type = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            insurance_type = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            facility_type = np.random.choice([0, 1], p=[0.7, 0.3])
            admission_day = visit
            length_of_stay = min(30, max(1, 
                np.random.poisson(3) + 2 * (admission_type == 2)))
            
            # Generate outcome based on clinical factors with some spurious correlation
            clinical_risk = (
                0.4 * (heart_rate > 100) +
                0.5 * (oxygen_saturation < 95) +
                0.3 * (temperature > 38) +
                0.3 * (blood_pressure > 140) +
                0.2 * health_conditions['diabetes'][patient_id]
            )
            
            admin_bias = (
                0.2 * (insurance_type == 0) +  # Public insurance
                0.15 * (admission_type == 2) +  # Emergency admission
                0.1 * (facility_type == 1)  # Urban hospital
            )
            
            outcome = int((clinical_risk + admin_bias * 0.3) > 0.5)
            
            data.append({
                'patient_id': patient_id,
                'visit': visit,
                'heart_rate': heart_rate,
                'blood_pressure': blood_pressure,
                'temperature': temperature,
                'respiratory_rate': respiratory_rate,
                'oxygen_saturation': oxygen_saturation,
                'lab_values': lab_values,
                'medications': medications,
                'admission_type': admission_type,
                'insurance_type': insurance_type,
                'facility_type': facility_type,
                'admission_day': admission_day,
                'length_of_stay': length_of_stay,
                'outcome': outcome,
                'has_hypertension': int(health_conditions['hypertension'][patient_id]),
                'has_diabetes': int(health_conditions['diabetes'][patient_id]),
                'has_copd': int(health_conditions['copd'][patient_id])
            })
    
    return pd.DataFrame(data)

def save_datasets(financial_data: pd.DataFrame, healthcare_data: pd.DataFrame) -> None:
    """Save generated datasets to CSV files"""
    financial_data.to_csv('synthetic_financial_data.csv', index=False)
    healthcare_data.to_csv('synthetic_healthcare_data.csv', index=False)
    print("Datasets saved to synthetic_financial_data.csv and synthetic_healthcare_data.csv")

if __name__ == "__main__":
    # Generate example datasets
    print("Generating synthetic financial data...")
    financial_data = generate_financial_data()
    
    print("Generating synthetic healthcare data...")
    healthcare_data = generate_healthcare_data()
    
    # Save datasets
    save_datasets(financial_data, healthcare_data)
    
    # Print samples
    print("\nFinancial Data Sample:")
    print(financial_data.head())
    
    print("\nHealthcare Data Sample:")
    print(healthcare_data.head())