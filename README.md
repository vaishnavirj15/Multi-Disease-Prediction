#Disease Prediction Platform
Overview
This interactive web-based platform, built using Streamlit, enables users to assess the risk of certain diseases based on their health metrics. The application predicts the likelihood of diabetes, stroke, or heart disease using machine learning models trained on publicly available datasets.
With a focus on accuracy and usability, the platform employs Random Forest and Logistic Regression models, achieving up to 90% accuracy for stroke prediction.

Key Features
Multi-Disease Support: Choose from Diabetes, Stroke, or Framingham Heart Disease risk prediction.
Interactive Input Forms: Enter personalized health metrics for real-time predictions.
Accurate Predictions: Models are fine-tuned for high accuracy using preprocessing techniques and scalable algorithms.
Evaluation Metrics: Visualize performance with confusion matrices and other key metrics.
Technologies Used
Languages: Python
Frameworks: Streamlit, Scikit-learn
Libraries: Pandas, Matplotlib, Seaborn, Imbalanced-Learn
Development: Google Colab

Setup Instructions
Pre-requisites
a.Ensure the following are installed on your system:

Python 3.7 or higher
Google Colab (for local or online development)
Installation
a.Clone this repository:
  git clone https://github.com/your-username/disease-predictor.git  
  cd disease-predictor 
b.Install the required libraries:
  pip install streamlit pandas matplotlib seaborn scikit-learn imbalanced-learn  
c.Mount Google Drive (for accessing datasets in Colab):
  from google.colab import drive  
  drive.mount('/content/drive') 
