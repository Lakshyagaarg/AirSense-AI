# AirAenseAI: Air Quality Index (AQI) Prediction

A machine learning-powered web application for predicting air quality based on pollutant concentrations.

## Features

- **Real-time AQI Predictions**: Get AQI values and categories instantly
- **Interactive Parameter Input**: Adjust pollutant levels with intuitive sliders
- **Multi-Model Support**: 
  - Random Forest Classifier for AQI bucket classification (Good → Severe)
  - Ridge Regression for precise AQI value prediction
- **Feature Information**: Learn about 9 major air pollutants
- **Model Performance Metrics**: View accuracy, precision, recall, and other metrics
- **Health Recommendations**: Get personalized advice based on AQI levels

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Streamlit 1.0+
- scikit-learn
- pandas
- numpy

### Installation

# Install required packages
pip install streamlit pandas numpy scikit-learn

# Or install from requirements
pip install -r requirements.txt

### Running the App
# From the project directory
streamlit run app.py

The app will open in your default browser at `http://localhost:8501`

## How to Use

1. **Navigate to the Prediction Tab** (default)
2. **Adjust Pollutant Levels** using the sliders:
   - PM2.5 and PM10: Fine and coarse particulate matter
   - NO, NO2, NOx: Nitrogen oxides
   - NH3: Ammonia
   - CO: Carbon monoxide
   - SO2: Sulfur dioxide
   - O3: Ozone

3. **View Results**:
   - AQI Value (Regression): Numerical AQI score
   - AQI Bucket (Classification): Category (Good → Severe)
   - Interpretation: Health impact and recommendations

4. **Explore Feature Info** to learn about pollutants
5. **Check Model Performance** to see model metrics

## Models Used

### Classification (Random Forest Classifier)
- **Predicts**: AQI Bucket (0-5 scale)
  - 0: Good
  - 1: Satisfactory
  - 2: Moderate
  - 3: Poor
  - 4: Very Poor
  - 5: Severe

### Regression (Ridge Regression)
- **Predicts**: Exact AQI value (0-500+)
- Trained on standardized features

### Feature Scaling
- StandardScaler: Normalizes all pollutant inputs to zero mean and unit variance

## Project Structure

```
AQI Project/
├── app.py                          # Main Streamlit app
├── test_app.py                     # Test script for validation
├── Notebook.ipynb                  # Jupyter notebook with full analysis
├── Data/
│   ├── city_day.csv.zip
│   └── station_day.csv.csv.zip
├── Pickle_Files/
│   ├── best_model.pkl              # Random Forest model
│   ├── ridge_model.pkl             # Ridge Regression model
│   └── scaler.pkl                  # StandardScaler
├── Result_csv/
│   ├── model_results.csv
│   ├── Lasso_model_results.csv
│   └── classification_results.csv
└── README.md                       # This file
```

## Input Ranges

| Pollutant | Range (µg/m³) |
|-----------|---------------|
| PM2.5     | 0 - 500       |
| PM10      | 0 - 500       |
| NO        | 0 - 200       |
| NO2       | 0 - 200       |
| NOx       | 0 - 300       |
| NH3       | 0 - 100       |
| CO        | 0 - 10        |
| SO2       | 0 - 100       |
| O3        | 0 - 300       |

## AQI Categories

| AQI Range | Category   | Health Impact                            |
|-----------|------------|------------------------------------------|
| 0-50      | Good       | No impact                                |
| 51-100    | Satisfactory | Acceptable; some may be concerned      |
| 101-200   | Moderate   | Some groups may experience effects      |
| 201-300   | Poor       | Sensitive groups affected                |
| 301-400   | Very Poor  | Most people may experience effects      |
| >400      | Severe     | Health warning; avoid outdoors          |

## Model Performance

### Classification Models
Model               | Accuracy | Precision | Recall | F1 Score
--------------------|----------|-----------|--------|----------
Logistic Regression | ~0.75    | ~0.75     | ~0.75  | ~0.74
Random Forest       | ~0.85    | ~0.85     | ~0.85  | ~0.84

### Regression Models
Model              | MAE    | RMSE   | R² Score | MSE
-------------------|--------|--------|----------|------
Linear Regression  | ~18    | ~25    | ~0.68    | ~625
Ridge Regression   | ~18    | ~25    | ~0.68    | ~625
Lasso Regression   | ~20    | ~28    | ~0.63    | ~784

## Configuration

Model paths are configured in `app.py`:
- Pickle files: `Pickle_Files/`
- Result CSVs: `Result_csv/`

All paths use absolute Windows paths. Update paths in the code if moving the project.

## Troubleshooting

### "Models not found" error
- Ensure `Pickle_Files/` directory contains all three pickle files
- Check that paths in `app.py` match your system

### "Streamlit not found"
pip install streamlit>=1.0


### App runs slowly
- First load caches data: wait for "Running..." to complete
- Subsequent predictions are instant due to caching

## Notes

- The app uses `@st.cache_resource` for model loading (loaded once per session)
- The app uses `@st.cache_data` for result CSV loading
- Default slider values are set at ~30% of max range

## Contributing

To modify or extend the app:
1. Update `app.py` with new features
2. Test with `test_app.py`
3. Ensure backward compatibility with trained models

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Models](https://scikit-learn.org/)
- [Air Quality Standards](https://www.epa.gov/air-quality)

## License

This project is part of an Air Quality Index prediction analysis.


**Created:** February 2025  
**Last Updated:** February 2025
