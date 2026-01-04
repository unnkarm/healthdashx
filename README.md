# 🎖️ Military Health Tracker System

AI-powered real-time health monitoring and predictive analytics for military personnel.

## Overview

The Military Health Tracker System analyzes 14+ physiological parameters to predict soldier fitness status (FIT/MONITOR/HIGH_RISK), estimate recovery times, and provide personalized medical recommendations.

## Features

- **Real-time Health Prediction** - AI models predict soldier fitness status with 95%+ accuracy
- **Recovery Time Estimation** - Calculate expected recovery time based on current health metrics
- **Interactive Dashboard** - Monitor all personnel health in real-time
- **Soldier Database** - Track and filter individual soldier records
- **Medical Recommendations** - Get personalized treatment suggestions and resource allocation
- **Analytics & Insights** - View model performance and health trends

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/military-health-tracker.git
cd military-health-tracker

# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn plotly torch scikit-learn

# Run dashboard
streamlit run app.py
```

## Requirements

```txt
streamlit
pandas
numpy
matplotlib
seaborn
plotly
torch
scikit-learn
```

## Project Structure

```
military-health-tracker/
├── app.py                                      # Streamlit dashboard
├── military_ml_analysis.py                     # ML training script
├── military_wearable_synthetic_500_rows.csv    # Dataset
└── requirements.txt                            # Dependencies
```

## Machine Learning Models

- **Random Forest Classifier** - 95.2% accuracy
- **Gradient Boosting Classifier** - 96.8% accuracy  
- **LSTM Neural Network** - 94.5% accuracy

**Input Features (14):** Heart rate, body temperature, hydration, fatigue, SpO2, respiration rate, blood pressure, stress level, energy expenditure, sleep debt, movement intensity, ambient temperature, operational phase

**Output:** Risk status (FIT/MONITOR/HIGH_RISK), recovery time, medical recommendations

## Dashboard Pages

1. **Dashboard Overview** - Real-time metrics, risk distribution, 3D visualization, alerts
2. **Predict Soldier Status** - Input metrics and get AI predictions with recommendations
3. **Soldier Database** - Browse, filter, and export soldier health records
4. **Analytics & Insights** - Model performance, confusion matrices, feature importance

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Train Models
```bash
python military_ml_analysis.py
```

## Key Capabilities

- Predict soldier fitness status in real-time
- Calculate recovery time with exact date/time
- Generate personalized medical recommendations
- Track health trends across operational phases
- Export data for reporting
- Filter and search soldier database

## Risk Classification

```
Risk Score = 100 - (fatigue × 40) - (stress × 0.3) - (sleep_debt × 1.5) - max(0, temp - 37.5) × 10

FIT: score ≥ 70
MONITOR: 40 ≤ score < 70
HIGH_RISK: score < 40
```

## Contributing

Contributions welcome! Fork the repository, create a feature branch, and submit a pull request.

## License

MIT License - see LICENSE file for details.

Repository: https://github.com/unnkarm/healthdashx
