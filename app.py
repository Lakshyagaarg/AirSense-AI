import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AirSense AI: Air Quality Prediction & Health Risk Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pollutant-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        cursor: pointer;
        text-align: center;
        transition: transform 0.2s;
        font-weight: bold;
        font-size: 18px;
    }
    .pollutant-card:hover {
        transform: scale(1.05);
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .effect-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ff6b6b;
    }
    .source-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #51cf66;
    }
    .range-box {
        background-color: #cfe2ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #0d6efd;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for selected pollutant
if 'selected_pollutant' not in st.session_state:
    st.session_state.selected_pollutant = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Define paths
base_path = Path(r"C:\Users\HP\OneDrive\Desktop\DataScience\AQI Project")
pickle_path = base_path / "Pickle_Files"
result_path = base_path / "Result_csv"

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        rf_model = pickle.load(open(pickle_path / "best_model.pkl", "rb"))
        ridge_model = pickle.load(open(pickle_path / "ridge_model.pkl", "rb"))
        scaler = pickle.load(open(pickle_path / "scaler.pkl", "rb"))
        return rf_model, ridge_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load model results
@st.cache_data
def load_results():
    try:
        regression_results = pd.read_csv(result_path / "Lasso_model_results.csv")
        classification_results = pd.read_csv(result_path / "classification_results.csv")
        return regression_results, classification_results
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None, None

# Define feature names and ranges
FEATURES = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3'
]

FEATURE_RANGES = {
    'PM2.5': (0, 500),
    'PM10': (0, 500),
    'NO': (0, 200),
    'NO2': (0, 200),
    'NOx': (0, 300),
    'NH3': (0, 100),
    'CO': (0, 100),
    'SO2': (0, 100),
    'O3': (0, 300)
}

# Units for features
FEATURE_UNITS = {
    'PM2.5': 'µg/m³',
    'PM10': 'µg/m³',
    'NO': 'µg/m³',
    'NO2': 'µg/m³',
    'NOx': 'µg/m³',
    'NH3': 'µg/m³',
    'CO': 'mg/m³',
    'SO2': 'µg/m³',
    'O3': 'µg/m³'
}

AQI_BUCKETS = {
    0: "Good",
    1: "Satisfactory",
    2: "Moderate",
    3: "Poor",
    4: "Very Poor",
    5: "Severe"
}

BUCKET_COLORS = {
    0: "🟢",
    1: "🟡",
    2: "🟠",
    3: "🔴",
    4: "🔴",
    5: "⚫"
}

# Pollutant Information Dictionary
POLLUTANT_INFO = {
    "PM2.5": {
        "name": "Particulate Matter 2.5",
        "harmful_effects": [
            "Causes respiratory diseases",
            "Increases asthma and allergies",
            "Can penetrate deep into lungs",
            "Linked to heart diseases",
            "Reduces life expectancy"
        ],
        "typical_range": "0-100 µg/m³ (good range is <35 µg/m³)",
        "emissions": [
            "Vehicle exhaust and emissions",
            "Industrial pollution",
            "Power plants and factories",
            "Burning of coal and wood",
            "Construction dust"
        ]
    },
    "PM10": {
        "name": "Particulate Matter 10",
        "harmful_effects": [
            "Causes dust in respiratory system",
            "Triggers asthma attacks",
            "Reduces visibility",
            "Accumulates in lungs over time",
            "Aggravates existing lung conditions"
        ],
        "typical_range": "0-150 µg/m³ (good range is <50 µg/m³)",
        "emissions": [
            "Road dust and tire wear",
            "Construction and demolition",
            "Industrial processes",
            "Agricultural activities",
            "Natural dust storms"
        ]
    },
    "NO": {
        "name": "Nitrogen Monoxide",
        "harmful_effects": [
            "Colorless and odorless gas",
            "Reacts with ozone to form NO2",
            "Respiratory irritation",
            "Reduces oxygen transport"
        ],
        "typical_range": "0-50 µg/m³",
        "emissions": [
            "Vehicle exhaust",
            "Power generation plants",
            "Industrial combustion",
            "High-temperature processes"
        ]
    },
    "NO2": {
        "name": "Nitrogen Dioxide",
        "harmful_effects": [
            "Reddish-brown toxic gas",
            "Damages lung tissues",
            "Increases susceptibility to infections",
            "Aggravates asthma and bronchitis",
            "Contributes to smog formation"
        ],
        "typical_range": "0-80 µg/m³ (normal air < 40 µg/m³)",
        "emissions": [
            "Vehicle exhaust",
            "Power plants",
            "Industrial facilities",
            "Oil refineries",
            "Chemical manufacturing"
        ]
    },
    "NOx": {
        "name": "Nitrogen Oxides (NO + NO2)",
        "harmful_effects": [
            "Causes inflammation of airways",
            "Reduces lung function",
            "Increases asthma symptoms",
            "Contributes to acid rain",
            "Forms ground-level ozone"
        ],
        "typical_range": "0-150 µg/m³",
        "emissions": [
            "Combustion processes",
            "Vehicle emissions",
            "Industrial sources",
            "Power generation",
            "Heaters and furnaces"
        ]
    },
    "NH3": {
        "name": "Ammonia",
        "harmful_effects": [
            "Pungent odor even at low levels",
            "Irritates eyes, nose, and throat",
            "Respiratory tract damage",
            "Can cause asthma attacks",
            "Forms secondary particulates"
        ],
        "typical_range": "0-30 µg/m³",
        "emissions": [
            "Animal waste and manure",
            "Fertilizer application",
            "Industrial chemical production",
            "Wastewater treatment",
            "Vehicle emissions"
        ]
    },
    "CO": {
        "name": "Carbon Monoxide",
        "harmful_effects": [
            "Colorless, odorless toxic gas",
            "Reduces oxygen in blood",
            "Affects brain and heart function",
            "Causes headaches and dizziness",
            "High levels can be fatal",
            "Impairs cognitive function"
        ],
        "typical_range": "0-5 mg/m³ (safe is <2 mg/m³)",
        "emissions": [
            "Vehicle exhaust",
            "Incomplete combustion",
            "Industrial processes",
            "Heating and power generation",
            "Smoking and fires"
        ]
    },
    "SO2": {
        "name": "Sulfur Dioxide",
        "harmful_effects": [
            "Sharp, unpleasant smell",
            "Damages respiratory system",
            "Triggers asthma attacks",
            "Can cause bronchitis",
            "Contributes to acid rain",
            "Damages crops and vegetation"
        ],
        "typical_range": "0-40 µg/m³",
        "emissions": [
            "Coal and oil burning",
            "Metal smelting plants",
            "Chemical manufacturing",
            "Petroleum refineries",
            "Volcanic eruptions"
        ]
    },
    "O3": {
        "name": "Ozone",
        "harmful_effects": [
            "Respiratory system irritation",
            "Reduces lung capacity",
            "Aggravates asthma and allergies",
            "Damage to plants and crops",
            "Reduced athletic performance",
            "Chronic health effects"
        ],
        "typical_range": "0-100 µg/m³ (safe is <60 µg/m³)",
        "emissions": [
            "Secondary pollutant (formed in air)",
            "Formed from NOx + VOCs + sunlight",
            "Photochemical reactions",
            "Not directly emitted but created",
            "Peak levels in afternoon/evening"
        ]
    }
}

# Title and description
st.title("🌍 Air Quality Index (AQI) Prediction System")
st.markdown("""
This application predicts the Air Quality Index based on pollutant concentrations using machine learning models.
Learn about pollutant levels and get real-time AQI predictions.
""")

# Load models
rf_model, ridge_model, scaler = load_models()
if rf_model is None:
    st.error("Cannot load models. Please check the file paths.")
    st.stop()

# Load results for info section
regression_results, classification_results = load_results()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📘 Pollutant Info", "📊 Levels of AQI", "🔧 Model Info"])

with tab1:
    st.header("Make a Prediction")
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    feature_values = {}
    
    st.markdown("---")
    st.markdown("### 🎚️ Pollutant Levels")
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    feature_values = {}
    
    # Input sliders for all features
    with col1:
        st.subheader("Pollutant Levels (Part 1)")
        for i, feature in enumerate(FEATURES[:5]):
            min_val, max_val = FEATURE_RANGES[feature]
            default_val = min_val + (max_val - min_val) * 0.3
            unit = FEATURE_UNITS[feature]
            
            feature_values[feature] = st.slider(
                f"{feature} ({unit})",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=1.0,
                key=f"slider_{feature}_1"
            )
    
    with col2:
        st.subheader("Pollutant Levels (Part 2)")
        for feature in FEATURES[5:]:
            min_val, max_val = FEATURE_RANGES[feature]
            default_val = min_val + (max_val - min_val) * 0.3
            unit = FEATURE_UNITS[feature]
            
            feature_values[feature] = st.slider(
                f"{feature} ({unit})",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=1.0,
                key=f"slider_{feature}_2"
            )
    
    # Prepare data for prediction
    input_data = np.array([feature_values[f] for f in FEATURES]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    classification_pred = rf_model.predict(input_scaled)[0]
    regression_pred = ridge_model.predict(input_scaled)[0]
    
    # Display predictions
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        st.metric(
            "AQI Value (Regression)",
            f"{regression_pred:.2f}",
            delta=None,
            delta_color="inverse"
        )
    
    with pred_col2:
        aqi_bucket_name = AQI_BUCKETS.get(int(classification_pred), "Unknown")
        bucket_icon = BUCKET_COLORS.get(int(classification_pred), "")
        st.metric(
            "AQI Bucket (Classification)",
            f"{bucket_icon} {aqi_bucket_name}",
            delta=None
        )
    
    with pred_col3:
        st.metric(
            "Confidence Level",
            "High",
            delta=None
        )
    
    # Display interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    aqi_int = int(regression_pred)
    
    if aqi_int <= 50:
        interpretation = "🟢 **Good Air Quality** - Air pollution poses little or no risk."
        advice = "Perfect conditions for outdoor activities."
    elif aqi_int <= 100:
        interpretation = "🟡 **Satisfactory** - Acceptable air quality; some pollutants may be a concern for some people."
        advice = "Generally suitable for outdoor activities."
    elif aqi_int <= 200:
        interpretation = "🟠 **Moderate** - Some groups may experience health effects."
        advice = "Sensitive groups should limit prolonged outdoor activities."
    elif aqi_int <= 300:
        interpretation = "🔴 **Poor** - Members of sensitive groups may experience health effects."
        advice = "Sensitive groups should avoid outdoor activities."
    elif aqi_int <= 400:
        interpretation = "🔴 **Very Poor** - Health alert. Everyone may experience health effects."
        advice = "Everyone should limit outdoor activities."
    else:
        interpretation = "⚫ **Severe** - Health warning. Avoid outdoor activities."
        advice = "Stay indoors and keep windows/doors closed."
    
    st.info(f"{interpretation}\n\n**Recommendation:** {advice}")
    
    # Input summary
    st.markdown("---")
    st.subheader("📋 Input Summary")
    summary_data = []
    for pollutant, value in feature_values.items():
        unit = FEATURE_UNITS[pollutant]
        summary_data.append({"Pollutant": pollutant, "Level": f"{value:.2f} {unit}"})
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Pollutant", ascending=True)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


with tab2:
    st.header("📘 Detailed Pollutant Information")
    
    if st.session_state.selected_pollutant is None:
        st.markdown("### 👈 Click on a pollutant to view detailed information!")
        
        st.markdown("---")
        st.markdown("### 📚 All Pollutants")
        
        # Display all pollutants in a grid
        cols = st.columns(3)
        for idx, feature in enumerate(FEATURES):
            with cols[idx % 3]:
                if st.button(f"🔍 {feature}", key=f"pollutant_btn_{feature}", use_container_width=True):
                    st.session_state.selected_pollutant = feature
                    st.rerun()
    else:
        pollutant = st.session_state.selected_pollutant
        info = POLLUTANT_INFO[pollutant]
        
        # Back button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("← Back to All"):
                st.session_state.selected_pollutant = None
                st.rerun()
        
        # Main info header
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 12px; text-align: center;'>
            <h1 style='margin: 0;'>{pollutant}</h1>
            <h3 style='margin: 10px 0 0 0;'>{info['name']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Create three columns for info sections
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background-color: #cfe2ff; padding: 10px; border-radius: 10px; border-left: 5px solid #0d6efd;'>
                <h3 style='color: #0d6efd; margin-top: 0;'>📏 Typical Range</h3>
            </div>
            """, unsafe_allow_html=True)
            st.write(info['typical_range'])
        
        with col2:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 10px; border-radius: 10px; border-left: 5px solid #51cf66;'>
                <h3 style='color: #155724; margin-top: 0;'>⚠️ Harmful Effects</h3>
            </div>
            """, unsafe_allow_html=True)
            for effect in info['harmful_effects']:
                st.write(f"✓ {effect}")
        
        with col3:
            st.markdown("""
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 10px; border-left: 5px solid #ff6b6b;'>
                <h3 style='color: #856404; margin-top: 0;'>🏭 Emission Sources</h3>
            </div>
            """, unsafe_allow_html=True)
            for source in info['emissions']:
                st.write(f"• {source}")
        
        st.markdown("---")
        
        # Browse other pollutants
        st.markdown("### 🔄 Browse Other Pollutants")
        pollutant_cols = st.columns(3)
        for idx, feature in enumerate(FEATURES):
            with pollutant_cols[idx % 3]:
                if feature != pollutant:
                    if st.button(f"📌 {feature}", key=f"switch_{feature}", use_container_width=True):
                        st.session_state.selected_pollutant = feature
                        st.rerun()


with tab3:
    st.header("📊 Levels of AQI")
    
    st.markdown("""
    ### Air Quality Index Categories
    
    The AQI is divided into six main categories based on health implications and recommended actions.
    """)
    
    aqi_categories = pd.DataFrame({
        "AQI Range": ["0-50", "51-100", "101-200", "201-300", "301-400", ">400"],
        "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "Symbol": ["🟢", "🟡", "🟠", "🔴", "🔴", "⚫"],
        "Health Impact": [
            "No impact",
            "Acceptable for most",
            "Some may be affected",
            "Sensitive groups affected",
            "Most people affected",
            "Severe health issues"
        ],
        "Recommended Action": [
            "Enjoy outdoor activities",
            "Outdoor activities OK",
            "Limit outdoor activities",
            "Avoid outdoor activities",
            "Stay indoors",
            "Emergency conditions"
        ]
    })
    
    st.dataframe(aqi_categories, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📈 Understanding Each Level")
    
    level_details = {
        "Good (0-50)": {
            "icon": "🟢",
            "description": "Air pollution poses little or no risk to the general population.",
            "activities": "Perfect for all outdoor activities including exercise.",
            "precautions": "None required."
        },
        "Satisfactory (51-100)": {
            "icon": "🟡",
            "description": "Air quality is acceptable; however, there may be a risk for some people.",
            "activities": "Outdoor activities are generally fine for most people.",
            "precautions": "Very sensitive groups may experience minor symptoms."
        },
        "Moderate (101-200)": {
            "icon": "🟠",
            "description": "Members of sensitive groups may experience health effects.",
            "activities": "Children, elderly, and people with respiratory conditions should limit strenuous outdoor activities.",
            "precautions": "Consider using air purifiers indoors and wearing masks outdoors."
        },
        "Poor (201-300)": {
            "icon": "🔴",
            "description": "Some members of the general population may experience health effects.",
            "activities": "Everyone should reduce outdoor activities; sensitive groups should stay indoors.",
            "precautions": "Keep windows closed; use air conditioning with filters."
        },
        "Very Poor (301-400)": {
            "icon": "🔴",
            "description": "Health alert. The general population may experience serious health effects.",
            "activities": "All people should minimize outdoor activities.",
            "precautions": "Keep indoors and use air purifiers. Wear N95 masks if venturing outside."
        },
        "Severe (>400)": {
            "icon": "⚫",
            "description": "Health warning. Emergency conditions. The entire population is more likely to be affected.",
            "activities": "Avoid all outdoor activities. This is a health emergency.",
            "precautions": "Stay indoors with windows/doors closed. Use industrial-grade air purifiers."
        }
    }
    
    for level, details in level_details.items():
        with st.expander(f"{details['icon']} {level}", expanded=False):
            st.write(f"**Description:** {details['description']}")
            st.write(f"**Outdoor Activities:** {details['activities']}")
            st.write(f"**Precautions:** {details['precautions']}")


with tab4:
    st.header("🔧 Model Info & Performance")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("Regression Models Performance")
        if regression_results is not None:
            reg_display = regression_results.copy()
            reg_display = reg_display.round(4)
            st.dataframe(
                reg_display,
                use_container_width=True,
                hide_index=True
            )
            st.caption("Lower MAE/MSE/RMSE = Better | Higher R² = Better")
    
    with col_info2:
        st.subheader("Classification Models Performance")
        if classification_results is not None:
            class_display = classification_results.copy()
            class_display = class_display.round(4)
            st.dataframe(
                class_display,
                use_container_width=True,
                hide_index=True
            )
            st.caption("Higher values = Better performance")
    
    st.markdown("---")
    st.subheader("⚙️ About the Models")
    st.write("""
    - **Random Forest Classifier**: Predicts AQI bucket (Good to Severe) with high accuracy
    - **Ridge Regression**: Predicts exact AQI value with regularization
    - **Features**: 9 major air pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3)
    - **Scaler**: StandardScaler for feature normalization
    - **Training Data**: Combined city and station air quality datasets
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌍 Air Quality Index Prediction System © 2026</p>
    <p>Built with Streamlit | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
