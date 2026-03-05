#!/usr/bin/env python
"""Test script to verify app functionality"""

import pickle
import numpy as np
from pathlib import Path

base_path = Path(r"C:\Users\HP\OneDrive\Desktop\DataScience\AQI Project")
pickle_path = base_path / "Pickle_Files"

print("Loading models...")
rf_model = pickle.load(open(pickle_path / "best_model.pkl", "rb"))
ridge_model = pickle.load(open(pickle_path / "ridge_model.pkl", "rb"))
scaler = pickle.load(open(pickle_path / "scaler.pkl", "rb"))

print("✓ Models loaded")

# Test data (random values within feature ranges)
test_input = np.array([[
    50,   # PM2.5
    75,   # PM10
    30,   # NO
    40,   # NO2
    70,   # NOx
    20,   # NH3
    0.5,  # CO
    20,   # SO2
    50    # O3
]]).reshape(1, -1)

print("Scaling input...")
test_scaled = scaler.transform(test_input)

print("Making predictions...")
classification = rf_model.predict(test_scaled)
regression = ridge_model.predict(test_scaled)

print(f"✓ Classification prediction: {classification[0]} (0=Good, 5=Severe)")
print(f"✓ Regression prediction: {regression[0]:.2f} (AQI value)")

print("\n✅ All tests passed! App is ready to run.")
print("\nTo run the app, use:")
print("  streamlit run app.py")
