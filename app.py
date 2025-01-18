from random import randint
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.utils import parallel_backend
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

st.title("Disease Predictor")
selection = st.radio("Choose the Disease:", ["Diabetes", "Stroke", "Framingham Heart Disease"])

def plot_evaluation_metrics(y_test, y_pred, model_name):
    """Plot evaluation metrics for a given model and return the Matplotlib figure."""
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Display accuracy and classification report
    st.write(f"Accuracy: {acc:.2f}")
    st.write("Classification Report:\n", class_report)

if selection == "Diabetes":
    st.header("Diabetes Prediction")
    df = pd.read_csv("diabetes.csv")

    # Preprocessing Pipeline
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    smote = SMOTE(random_state=42)

    # Prepare features and labels
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Apply transformations
    X = imputer.fit_transform(X)
    X = poly.fit_transform(X)
    X = scaler.fit_transform(X)

    # Balance data with SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train RandomForestClassifier with RandomizedSearchCV
    rf = RandomForestClassifier(random_state=42)
    param_dist = {"n_estimators": [100, 200], "max_depth": [10, 20], "min_samples_split": [2, 5]}
    with parallel_backend("threading", n_jobs=-1):
        rf_search = RandomizedSearchCV(rf, param_dist, cv=3, n_iter=10, random_state=42)
        rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    # Evaluate the model
    y_pred = best_rf.predict(X_test)
    plot_evaluation_metrics(y_test, y_pred, "Diabetes Model")

    # User input for prediction
    st.write("Enter the details:")
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", value=0)
        glucose = st.number_input("Glucose", value=120)
        bp = st.number_input("BloodPressure", value=70)
        skin = st.number_input("SkinThickness", value=20)
    with col2:
        insulin = st.number_input("Insulin", value=30)
        bmi = st.number_input("BMI", value=25.0)
        dpf = st.number_input("DiabetesPedigreeFunction", value=0.5)
        age = st.number_input("Age", value=30)

    new_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    new_data = scaler.transform(poly.transform(new_data))

    if st.button("Predict Diabetes"):
        pred = best_rf.predict(new_data)
        st.write(f"Prediction: {'Positive' if pred[0] == 1 else 'Negative'}")

elif selection == "Stroke":
    st.header("Stroke Prediction")
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    df["bmi"].fillna(df["bmi"].mean(), inplace=True)
    df["smoking_status"].fillna("unknown", inplace=True)

    # Label encoding for categorical features
    gender_encoder = LabelEncoder().fit(df['gender'])
    marriage_encoder = LabelEncoder().fit(df['ever_married'])
    work_type_encoder = LabelEncoder().fit(df['work_type'])
    residence_encoder = LabelEncoder().fit(df['Residence_type'])
    smoking_encoder = LabelEncoder().fit(df['smoking_status'])

    df['gender'] = gender_encoder.transform(df['gender'])
    df['ever_married'] = marriage_encoder.transform(df['ever_married'])
    df['work_type'] = work_type_encoder.transform(df['work_type'])
    df['Residence_type'] = residence_encoder.transform(df['Residence_type'])
    df['smoking_status'] = smoking_encoder.transform(df['smoking_status'])

    # Scaling
    scaler = StandardScaler()
    df[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(df[['age', 'avg_glucose_level', 'bmi']])

    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']

    # Handling Class Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Hyperparameter tuning using RandomizedSearchCV (more efficient than GridSearchCV)
    param_dist = {
        'max_depth': [10, 20, 30, 40, 50],  # This is a list, not a single value
        'n_estimators': [100, 200, 300],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42, verbose=2)
    random_search.fit(X_train, y_train)

    # Get the best model
    best_rf = random_search.best_estimator_

    # Make predictions
    y_pred = best_rf.predict(X_test)

    # Calculate and display the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the model: {accuracy:.2f}")

    # Prediction interface
    st.write("Enter the details:")

    # Input form
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', gender_encoder.classes_)
        age = st.number_input('Age', min_value=0, max_value=100, value=30)
        hypertension = st.selectbox('Hypertension', [0, 1])
        heart_disease = st.selectbox('Heart Disease', [0, 1])
        ever_married = st.selectbox('Ever Married', marriage_encoder.classes_)

    with col2:
        work_type = st.selectbox('Work Type', work_type_encoder.classes_)
        residence_type = st.selectbox('Residence Type', residence_encoder.classes_)
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=500.0, value=100.0)
        bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
        smoking_status = st.selectbox('Smoking Status', smoking_encoder.classes_)

    # Prepare new data for prediction
    new_data = [[gender_encoder.transform([gender])[0], age, hypertension, heart_disease, marriage_encoder.transform([ever_married])[0],
                 work_type_encoder.transform([work_type])[0], residence_encoder.transform([residence_type])[0], avg_glucose_level, bmi,
                 smoking_encoder.transform([smoking_status])[0]]]

    # Perform prediction only when the button is clicked
    if st.button('Predict Stroke'):
        predictions = best_rf.predict(new_data)
        st.write(f"Prediction: {'Positive' if predictions[0] == 1 else 'Negative'}")
        plot_evaluation_metrics(y_test, y_pred, "Stroke Model")

elif selection == "Framingham Heart Disease":
    st.title('Framingham Heart Disease Prediction')
    df = pd.read_csv('framingham.csv')
    df.fillna(df.mean(), inplace=True)
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")

    st.write("Enter the details")
    col1, col2 = st.columns(2)
    with col1:
        male = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age')
        education = st.selectbox('Education', [1, 2, 3, 4])
        currentSmoker = st.selectbox('Current Smoker', [0, 1])
        cigsPerDay = st.number_input('Cigarettes per Day')
        BPMeds = st.selectbox('Blood Pressure Medication', [0, 1])
    with col2:
        prevalentStroke = st.selectbox('Prevalent Stroke', [0, 1])
        prevalentHyp = st.selectbox('Prevalent Hypertension', [0, 1])
        diabetes = st.selectbox('Diabetes', [0, 1])
        totChol = st.number_input('Total Cholesterol')
        sysBP = st.number_input('Systolic Blood Pressure')
        diaBP = st.number_input('Diastolic Blood Pressure')
        BMI = st.number_input('BMI')
        heartRate = st.number_input('Heart Rate')
        glucose = st.number_input('Glucose')

    new_data = [[1 if male == 'Male' else 0, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]]
    new_data_scaled = scaler.transform(new_data)
    if st.button('Predict Heart Disease'):
        predictions = model.predict(new_data_scaled)
        st.write(f"Prediction: {'Positive' if predictions[0] == 1 else 'Negative'}")

        # Show evaluation metrics
        plot_evaluation_metrics(y_test, y_pred, "Framingham_Model")
