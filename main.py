import numpy as np
import pandas as pd
from scipy.fftpack import dct
from sklearn.decomposition import PCA
import pywt

# Read the PMU data from a CSV file (replace 'pmu_data.csv' with your actual file path)
data = pd.read_csv('sample1.csv', header=None, names=['Timestamp', 'Type', 'Variable', 'Value'])

# Convert timestamp to a suitable format (if needed)
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ns')

# Pivot the data for better organization
pivoted_data = data.pivot_table(index=['Timestamp', 'Type'], columns='Variable', values='Value')

# Fill missing values if any (optional)
pivoted_data = pivoted_data.fillna(0)

# Define disturbance detection thresholds
threshold_v = 0.9  # Voltage threshold
threshold_f = 59.9  # Frequency threshold
threshold_df_dt = 0.1  # Rate of change of frequency threshold
 
# Define the number of PCA components to retain
num_pca_components = 5

# Define DCT compression threshold
dct_threshold = 0.1

# Function to detect disturbances
def detect_disturbance(data, threshold_v, threshold_f, threshold_df_dt):
    voltage = data['Voltage']
    frequency = data['Frequency']
    df_dt = np.gradient(frequency)
    
    disturbance_indices = np.where(
        (voltage >= threshold_v) |
        (frequency < threshold_f) |
        (df_dt > threshold_df_dt)
    )[0]
    
    return disturbance_indices

# Detect disturbances
disturbance_indices = detect_disturbance(pivoted_data, threshold_v, threshold_f, threshold_df_dt)

# Determine the window length
def determine_window_length(data, disturbance_indices):
    # Use statistical analysis or other methods to determine the window length
    # For simplicity, assume a fixed window length here
    window_length = 10  # 10 seconds
    return window_length

if disturbance_indices.any():
    window_length = determine_window_length(pivoted_data, disturbance_indices)
else:
    window_length = 10  # Default window length for normal conditions

# Apply PCA
def apply_pca(data, num_components):
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(data)
    return pca_result

# Apply DCT compression
def apply_dct_compression(data, threshold):
    compressed_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        signal = data[:, i]
        dct_coefficients = dct(signal, type=2, norm='ortho')
        significant_indices = np.where(np.abs(dct_coefficients) > threshold)[0]
        compressed_signal = np.zeros_like(signal)
        compressed_signal[significant_indices] = dct_coefficients[significant_indices]
        compressed_data[:, i] = compressed_signal
    return compressed_data

# Apply PCA
pca_result = apply_pca(pivoted_data.values, num_pca_components)

# Apply DCT compression
compressed_pmu_data = apply_dct_compression(pca_result, dct_threshold)

# Your compressed PMU data is now in 'compressed_pmu_data'
# You can further process or save it as needed
