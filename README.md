# 😴 Sleep Quality Prediction System

A machine learning web application that predicts sleep quality based on lifestyle, demographic, and health metrics. This system helps individuals understand how their daily habits impact sleep patterns.

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline for classifying sleep quality into categories like 'Excellent', 'Good', 'Average', and 'Poor'. The system analyzes various factors including physical activity, stress levels, daily steps, and demographic information to provide accurate sleep quality predictions.

## 🚀 Features

- **Data Preprocessing**: Automated handling of missing values, feature scaling, and categorical encoding
- **Multiple ML Models**: Comparison of 8 different classification algorithms
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Web Interface**: User-friendly Streamlit application for real-time predictions
- **Model Persistence**: Saved pipelines for easy deployment and inference

## 🛠️ Tech Stack

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, XGBoost, CatBoost
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Dill
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure
MACHINE_LEARNING_PROJECT/
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ └── predict_pipeline.py
│ ├── utils.py
│ ├── logger.py
│ └── exception.py
├── artifacts/
│ ├── train.csv
│ ├── test.csv
│ ├── model.pkl
│ ├── column_processor.pkl
│ └── label_encoder.pkl
├── app.py
└── requirements.txt


## 🎯 Model Performance

The system evaluates multiple algorithms and selects the best performing model based on accuracy:

- Random Forest Classifier
- Decision Tree Classifier  
- Gradient Boosting Classifier
- Logistic Regression
- K-Nearest Neighbors
- XGBoost Classifier
- CatBoost Classifier
- AdaBoost Classifier

## 📋 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Age of the individual (18-80 years) |
| Sleep Duration | Numerical | Hours of sleep per night (4.0-10.0 hours) |
| Physical Activity Level | Numerical | Activity level score (0-100) |
| Stress Level | Numerical | Self-reported stress (1-10 scale) |
| Daily Steps | Numerical | Number of steps per day (1000-20000) |
| Gender | Categorical | Male/Female |
| Occupation | Categorical | Profession/Job role |
| BMI Category | Categorical | Weight classification |
| Sleep Disorder | Categorical | Presence of sleep disorders |

🎮 Usage
Open the Streamlit web interface

Fill in your details in the input form:

Personal demographics

Lifestyle metrics

Health indicators

Click "Predict Sleep Quality"

View your predicted sleep quality category

📈 Results Interpretation
The model provides sleep quality predictions along with:

Accuracy scores for model evaluation

Classification reports for detailed performance metrics

Feature importance analysis (where applicable)

🔮 Future Enhancements
Integration with wearable device data

Real-time sleep tracking

Personalized sleep improvement recommendations

Mobile application development

Advanced deep learning models

🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📄 License
This project is licensed under the MIT License.

👨‍💻 Developer
Sushil Pal  &  Tanya Sharma 
Machine Learning Engineers
