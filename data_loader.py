import numpy as np

def generate_sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    data = np.sin(t) + np.random.normal(0, 0.1, size=len(t))
    return data

def load_data():
    """
    Load time series data.
    Replace this with your actual data loading logic (e.g., from CSV, API, etc.)
    """
    # For now, we'll use the sample data generator
    return generate_sample_data()

def load_test_data():
    """Load test data for evaluation"""
    np.random.seed(42)
    test_data = np.sin(np.linspace(10, 15, 500)) + np.random.normal(0, 0.1, 500)
    return test_data
