
# Multi-Disease Prediction Platform

This interactive web-based platform, built using **Streamlit**, enables users to assess the risk of certain diseases based on their health metrics. The application leverages machine learning models to predict the likelihood of **Diabetes**, **Stroke**, or **Framingham Heart Disease** based on user input. 

With a focus on accuracy and usability, the platform employs **Random Forest** and **Logistic Regression** models, achieving up to **90% accuracy** for stroke prediction. Addresses class imbalance through impelementing SMOTE.

---

## Key Features
- **Multi-Disease Support**: Choose from **Diabetes**, **Stroke**, or **Heart Disease** risk prediction.
- **Interactive Input Forms**: Enter personalized health metrics for real-time predictions.
- **Accurate Predictions**: Fine-tuned models for high accuracy using preprocessing techniques and scalable algorithms.
- **Evaluation Metrics**: Visualize performance with confusion matrices and other critical evaluation metrics.

---

## Technologies Used
- **Languages**: Python
- **Frameworks**: Streamlit, Scikit-learn
- **Libraries**: Pandas, Matplotlib, Seaborn, Imbalanced-Learn
- **Development Environment**: Google Colab

---

## Setup Instructions

### **Pre-requisites**
- Ensure the following are installed on your system:
  - Python 3.7 or higher  
  OR  
  - Use **Google Colab** for local or online development.

---

### **Installation**
1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/multi-disease-prediction.git
   cd multi-disease-prediction
2. **Install the required libraries**:
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn imbalanced-learn
3. **Prepare the dataset**:
   - **Download the datasets and upload them to your Google Drive (if using Colab)**.
   - **Update the code with the correct path to the datasets**.
### **Running the application**:
1. **Save the app code**:
   - **Ensure the main Streamlit code is saved as app.py in your working directory or Colab environment**.
2. **Start the Streamlit server**:
   - **If working locally, run**:
     ```bash
     streamlit run app.py
   - **For Colab users, run**:
     ```bash
     !streamlit run app.py & npx localtunnel --port 8501
3. **Access the Application**:
   - **Open the generated link from the localtunnel command in your browser to access the app**.
4. **Recent code**:
   - **app.py is the latest and improved version of the code**.

