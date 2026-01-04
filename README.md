🎖️ Military Health Tracker (HealthDashX)
An AI-powered health monitoring and risk prediction dashboard designed for military commanders and medical personnel to track soldier readiness in real-time.

📌 Overview
HealthDashX leverages machine learning to analyze physiological data from wearable sensors. It classifies soldier status into three categories—FIT, MONITOR, or HIGH RISK—while providing actionable medical recommendations and estimated recovery timelines.

✨ Key Features
Real-time Risk Prediction: Uses Random Forest and Gradient Boosting models (95%+ accuracy) to assess soldier health.

3D Health Space: Interactive visualization of the relationship between Heart Rate, Stress, and Fatigue.

Individual Diagnostics: Detailed breakdown of vital signs (SpO2, Core Temp, BP) for specific personnel.

Automated Recommendations: Generates medical advice and resource requirements (e.g., IV kits, medevac readiness) based on risk levels.

Operational Analytics: Insights into how different phases (Combat, Patrol, Training) impact unit-wide fatigue and stress.

🛠️ Tech Stack
Frontend: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn, PyTorch

Visualization: Plotly, Seaborn

🚀 Installation & Local Setup
Clone the repository:

Bash

git clone https://github.com/unnkarm/healthdashx.git
cd healthdashx
Create a virtual environment:

Bash

python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

streamlit run app.py
📊 Data Features
The model utilizes 14 key features for prediction, including:

Vitals: Heart Rate (BPM), SpO2 (%), Core Body Temp (°C), Respiration Rate.

Activity: Movement Intensity (g), Energy Expenditure (kcal/hr).

Wellness: Fatigue Index, Stress Level (0-100), Sleep Debt (hours), Hydration (%).

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contact
Project Lead: [Your Name/GitHub Profile]

Repository: https://github.com/unnkarm/healthdashx
