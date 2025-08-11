import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Sidebar navigation
with st.sidebar:
    st.title("Health Risk Prediction")
    selected = option_menu(
        menu_title="Select Prediction",
        options=["Home", "Diabetes", "Heart Disease", "Parkinson's", "Stroke", "Lung Cancer", "Chronic Kidney Disease"],
        icons=["house", "activity", "heart-pulse", "person-walking", "activity", "lungs", "droplet"]
    )

# Initialize models as None
diabetes_model = heart_model = parkinsons_model = stroke_model = lung_cancer_model = chronic_kidney_model = None
diabetes_features = heart_features = parkinsons_features = stroke_features = lung_cancer_features = chronic_kidney_features = None

# ===========================
# 1. Load and Train Diabetes Model
# ===========================
if os.path.exists('diabetes.csv'):
    try:
        diabetes_data = pd.read_csv('diabetes.csv')
        X = diabetes_data.drop(columns=['Outcome'])
        y = diabetes_data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        diabetes_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        diabetes_model.fit(X_train, y_train)
        diabetes_features = X.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Error loading diabetes model: {str(e)}")

# ===========================
# 2. Load and Train Heart Disease Model
# ===========================
if os.path.exists('heart.csv'):
    try:
        heart_data = pd.read_csv('heart.csv')
        X = heart_data.drop(columns=['target'])
        y = heart_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        heart_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        heart_model.fit(X_train, y_train)
        heart_features = X.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Error loading heart disease model: {str(e)}")

# ===========================
# 3. Load and Train Parkinson's Model
# ===========================
if os.path.exists('parkinsons.csv'):
    try:
        parkinsons_data = pd.read_csv('parkinsons.csv')
        X = parkinsons_data.drop(columns=['name', 'status'])
        y = parkinsons_data['status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        parkinsons_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        parkinsons_model.fit(X_train, y_train)
        parkinsons_features = X.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Error loading Parkinson's model: {str(e)}")

# ===========================
# 4. Load and Train Stroke Model
# ===========================
if os.path.exists('stroke-data.csv'):
    try:
        stroke_data = pd.read_csv('stroke-data.csv')
        # Handle missing values
        stroke_data['bmi'] = pd.to_numeric(stroke_data['bmi'], errors='coerce')
        stroke_data['bmi'].fillna(stroke_data['bmi'].median(), inplace=True)
        
        # Convert categorical variables
        stroke_data['ever_married'] = stroke_data['ever_married'].map({'Yes': 1, 'No': 0})
        
        # One-hot encode categorical variables
        stroke_data_encoded = pd.get_dummies(stroke_data, 
                                           columns=['work_type', 'Residence_type', 'smoking_status'],
                                           drop_first=True)
        
        # Prepare features and target
        X = stroke_data_encoded.drop(columns=['id', 'stroke'])
        y = stroke_data_encoded['stroke']
        
        # Store feature columns for later use
        stroke_features = X.columns.tolist()
        
        # Split and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        stroke_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        stroke_model.fit(X_train, y_train)
    except Exception as e:
        st.sidebar.error(f"Error loading stroke model: {str(e)}")

# ===========================
# 5. Load and Train Lung Cancer Model
# ===========================
if os.path.exists('lung_cancer.csv'):
    try:
        lung_cancer_data = pd.read_csv('lung_cancer.csv')
        # Drop non-numeric columns
        lung_cancer_data = lung_cancer_data.drop(columns=['Name', 'Surname'])
        
        # Convert target to binary
        lung_cancer_data['Result'] = lung_cancer_data['Result'].apply(
            lambda x: 1 if str(x).lower() in ['1', 'true', 'yes'] else 0
        )
        
        X = lung_cancer_data.drop(columns=['Result'])
        y = lung_cancer_data['Result']
        
        # Store feature columns for later use
        lung_cancer_features = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lung_cancer_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lung_cancer_model.fit(X_train, y_train)
    except Exception as e:
        st.sidebar.error(f"Error loading lung cancer model: {str(e)}")

# ===========================
# 6. Load and Train Chronic Kidney Disease Model
# ===========================
if os.path.exists('ChronicKidneyDisease_EHRs_from_AbuDhabi.csv'):
    try:
        ckd_data = pd.read_csv('ChronicKidneyDisease_EHRs_from_AbuDhabi.csv')
        
        # Check if required columns exist
        if 'EventCKD35' not in ckd_data.columns:
            st.sidebar.error("Required column 'EventCKD35' not found in the dataset.")
        else:
            # Handle missing values
            numeric_cols = ckd_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col != 'EventCKD35':  # Don't fill the target variable
                    ckd_data[col] = pd.to_numeric(ckd_data[col], errors='coerce')
                    ckd_data[col].fillna(ckd_data[col].median(), inplace=True)
            
            # Prepare features and target
            X = ckd_data.drop(columns=['EventCKD35'])
            y = ckd_data['EventCKD35']
            
            # Store feature columns for later use
            chronic_kidney_features = X.columns.tolist()
            
            # Split and train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            chronic_kidney_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            chronic_kidney_model.fit(X_train, y_train)
    except Exception as e:
        st.sidebar.error(f"Error loading Chronic Kidney Disease model: {str(e)}")

# ===========================
# Home Page
# ===========================
if selected == "Home":
    st.title("Welcome to Health Risk Prediction App")
    st.write("""
    This application helps predict various health risks using machine learning models.
    Please select a prediction type from the sidebar to get started.
    """)
    
    st.write("### Available Predictions:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Diabetes Prediction**
        - Predicts risk of diabetes
        - Based on health metrics
        """)
        
        st.info("""
        **Heart Disease Prediction**
        - Predicts risk of heart disease
        - Based on medical parameters
        """)
        
        st.info("""
        **Parkinson's Prediction**
        - Predicts Parkinson's disease
        - Based on voice measurements
        """)
    
    with col2:
        st.info("""
        **Stroke Prediction**
        - Predicts stroke risk
        - Based on health and lifestyle factors
        """)
        
        st.info("""
        **Lung Cancer Prediction**
        - Predicts lung cancer risk
        - Based on lifestyle factors
        """)
        
        st.info("""
        **Chronic Kidney Disease**
        - Predicts kidney disease risk
        - Based on medical test results
        """)
    
    st.write("---")
    st.warning("""
    **Disclaimer:** 
    This application is for informational purposes only and is not a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any 
    questions you may have regarding a medical condition.
    """)

# ===========================
# Diabetes Prediction Page
# ===========================
elif selected == "Diabetes":
    st.title("Diabetes Risk Prediction")
    st.write("Please enter the following health metrics:")
    
    if diabetes_model is None:
        st.error("Diabetes prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    # Input fields for diabetes prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, step=1)
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, step=1)
        
    with col2:
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, step=1)
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=1000, step=1)
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
        
    with col3:
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, step=0.01)
        age = st.number_input('Age (years)', min_value=0, max_value=120, step=1)
    
    if st.button('Check Diabetes Risk', type="primary", use_container_width=True):
        try:
            # Make prediction
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, diabetes_pedigree, age]])
            prediction = diabetes_model.predict_proba(input_data)[0][1]
            risk_percentage = round(prediction * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High Risk of Diabetes: {risk_percentage}%")
                st.write("""
                **Recommendations:**
                - Consult a healthcare professional
                - Monitor your blood sugar levels regularly
                - Follow a balanced diet and exercise routine
                """)
            else:
                st.success(f"Low Risk of Diabetes: {100 - risk_percentage}%")
                st.write("""
                **Maintain your health:**
                - Continue healthy eating habits
                - Exercise regularly
                - Get regular check-ups
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ===========================
# Heart Disease Prediction Page
# ===========================
elif selected == "Heart Disease":
    st.title("Heart Disease Risk Prediction")
    st.write("Please enter the following details:")
    
    if heart_model is None:
        st.error("Heart disease prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    # Input fields for heart disease prediction
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=250, step=1)
        chol = st.number_input('Cholesterol (mg/dl)', min_value=0, max_value=600, step=1)
        
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.selectbox('Resting Electrocardiographic Results', 
                              ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, step=1)
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, step=0.1)
    
    if st.button('Check Heart Disease Risk', type="primary", use_container_width=True):
        try:
            # Preprocess inputs
            sex_encoded = 1 if sex == 'Male' else 0
            cp_encoded = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
            fbs_encoded = 1 if fbs == 'Yes' else 0
            restecg_encoded = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
            exang_encoded = 1 if exang == 'Yes' else 0
            
            # Make prediction
            input_data = np.array([[age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                                  restecg_encoded, thalach, exang_encoded, oldpeak]])
            
            prediction = heart_model.predict_proba(input_data)[0][1]
            risk_percentage = round(prediction * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High Risk of Heart Disease: {risk_percentage}%")
                st.write("""
                **Recommendations:**
                - Consult a cardiologist
                - Monitor your blood pressure and cholesterol
                - Follow a heart-healthy diet and exercise plan
                """)
            else:
                st.success(f"Low Risk of Heart Disease: {100 - risk_percentage}%")
                st.write("""
                **Maintain your heart health:**
                - Continue healthy lifestyle habits
                - Exercise regularly
                - Get regular check-ups
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ===========================
# Parkinson's Prediction Page
# ===========================
elif selected == "Parkinson's":
    st.title("Parkinson's Disease Prediction")
    st.write("Please enter the following voice measurement details:")
    
    if parkinsons_model is None:
        st.error("Parkinson's prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    # Input fields for Parkinson's prediction
    col1, col2 = st.columns(2)
    
    with col1:
        mdvp_fo = st.number_input('MDVP:Fo(Hz)', min_value=50.0, max_value=300.0, step=0.1)
        mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', min_value=50.0, max_value=3000.0, step=0.1)
        mdvp_flo = st.number_input('MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, step=0.1)
        mdvp_jitter = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
        mdvp_jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, step=0.00001, format="%.5f")
        
    with col2:
        mdvp_rap = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
        mdvp_ppq = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
        jitter_ddp = st.number_input('Jitter:DDP', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
        mdvp_shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
        mdvp_shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    
    if st.button('Check Parkinson\'s Risk', type="primary", use_container_width=True):
        try:
            # Make prediction
            input_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs,
                                  mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db]])
            
            prediction = parkinsons_model.predict_proba(input_data)[0][1]
            risk_percentage = round(prediction * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High Risk of Parkinson's Disease: {risk_percentage}%")
                st.write("""
                **Recommendations:**
                - Consult a neurologist
                - Consider a comprehensive neurological examination
                - Discuss treatment options with a healthcare provider
                """)
            else:
                st.success(f"Low Risk of Parkinson's Disease: {100 - risk_percentage}%")
                st.write("""
                **Maintain your health:**
                - Continue regular check-ups
                - Monitor for any new symptoms
                - Maintain a healthy lifestyle
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ===========================
# Stroke Prediction Page
# ===========================
elif selected == "Stroke":
    st.title("Stroke Risk Assessment")
    st.write("Please provide the following information for stroke risk evaluation:")
    
    if stroke_model is None:
        st.error("Stroke prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    # Input fields for stroke prediction
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age (years)', min_value=0, max_value=120, step=1)
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
        heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
        ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
        work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
        
    with col2:
        residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
        avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0.0, max_value=300.0, step=0.1)
        bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0, step=0.1)
        smoking_status = st.selectbox('Smoking Status', 
                                    ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
    
    if st.button('Assess Stroke Risk', type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'age': [age],
                'hypertension': [1 if hypertension == 'Yes' else 0],
                'heart_disease': [1 if heart_disease == 'Yes' else 0],
                'ever_married': [1 if ever_married == 'Yes' else 0],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'work_type_' + work_type: [1],
                'Residence_type_' + residence_type: [1],
                'smoking_status_' + smoking_status.lower().replace(' ', '_'): [1]
            }
            
            # Create DataFrame and ensure all expected columns are present
            input_df = pd.DataFrame(input_data)
            
            # Add missing columns with 0
            for col in stroke_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[stroke_features]
            
            # Make prediction
            prediction = stroke_model.predict_proba(input_df)[0][1]
            risk_percentage = round(prediction * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High Risk of Stroke: {risk_percentage}%")
                st.write("""
                **Recommendations:**
                - Seek immediate medical attention if experiencing symptoms
                - Consult a healthcare provider for a comprehensive evaluation
                - Monitor and manage risk factors like blood pressure and cholesterol
                """)
            else:
                st.success(f"Low Risk of Stroke: {100 - risk_percentage}%")
                st.write("""
                **Maintain your health:**
                - Continue healthy lifestyle habits
                - Exercise regularly and maintain a healthy weight
                - Get regular check-ups
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ===========================
# Lung Cancer Prediction Page
# ===========================
elif selected == "Lung Cancer":
    st.title("Lung Cancer Risk Assessment")
    st.write("Please provide the following information for lung cancer risk evaluation:")
    
    if lung_cancer_model is None:
        st.error("Lung cancer prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    # Input fields for lung cancer prediction
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age (years)', min_value=0, max_value=120, step=1)
        smokes = st.number_input('Number of Cigarettes Smoked Daily', min_value=0, max_value=100, step=1)
        areaq = st.number_input('Area Q (lung function)', min_value=0, max_value=100, step=1)
        alkhol = st.number_input('Alcohol Consumption (standard drinks per week)', min_value=0, max_value=100, step=1)
        
    with col2:
        yellow_fingers = st.selectbox('Yellow Fingers', ['No', 'Yes'])
        anxiety = st.selectbox('Anxiety', ['No', 'Yes'])
        peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'])
        chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'])
    
    if st.button('Assess Lung Cancer Risk', type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Age': [age],
                'Smokes': [smokes],
                'AreaQ': [areaq],
                'Alkhol': [alkhol],
                'YELLOW_FINGERS': [1 if yellow_fingers == 'Yes' else 0],
                'ANXIETY': [1 if anxiety == 'Yes' else 0],
                'PEER_PRESSURE': [1 if peer_pressure == 'Yes' else 0],
                'CHRONIC DISEASE': [1 if chronic_disease == 'Yes' else 0]
            }
            
            # Create DataFrame and ensure all expected columns are present
            input_df = pd.DataFrame(input_data)
            
            # Add missing columns with 0
            for col in lung_cancer_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[lung_cancer_features]
            
            # Make prediction
            prediction = lung_cancer_model.predict_proba(input_df)[0][1]
            risk_percentage = round(prediction * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High Risk of Lung Cancer: {risk_percentage}%")
                st.write("""
                **Recommendations:**
                - Consult a healthcare provider for further evaluation
                - Consider a low-dose CT scan if recommended
                - Quit smoking and avoid secondhand smoke
                """)
            else:
                st.success(f"Low Risk of Lung Cancer: {100 - risk_percentage}%")
                st.write("""
                **Maintain your lung health:**
                - Avoid smoking and secondhand smoke
                - Get regular check-ups
                - Be aware of any persistent respiratory symptoms
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ===========================
# Chronic Kidney Disease Prediction Page
# ===========================
elif selected == "Chronic Kidney Disease":
    st.title("Chronic Kidney Disease Risk Assessment")
    st.write("Please provide the following medical information:")
    
    if chronic_kidney_model is None:
        st.error("Chronic Kidney Disease prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    # Input fields for CKD prediction
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age (years)', min_value=0, max_value=120, step=1)
        bp = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=250, step=1)
        sg = st.number_input('Specific Gravity', min_value=1.000, max_value=1.050, step=0.001, format="%.3f")
        al = st.number_input('Albumin (0-5)', min_value=0, max_value=5, step=1)
        su = st.number_input('Sugar (0-5)', min_value=0, max_value=5, step=1)
        
    with col2:
        rbc = st.selectbox('Red Blood Cells (normal/abnormal)', ['normal', 'abnormal'])
        pc = st.selectbox('Pus Cell (normal/abnormal)', ['normal', 'abnormal'])
        pcc = st.selectbox('Pus Cell Clumps (present/notpresent)', ['present', 'notpresent'])
        ba = st.selectbox('Bacteria (present/notpresent)', ['present', 'notpresent'])
        bgr = st.number_input('Blood Glucose Random (mg/dL)', min_value=0, max_value=500, step=1)
    
    if st.button('Assess Kidney Disease Risk', type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'age': [age],
                'bp': [bp],
                'sg': [sg],
                'al': [al],
                'su': [su],
                'rbc_normal': [1 if rbc == 'normal' else 0],
                'pc_normal': [1 if pc == 'normal' else 0],
                'pcc_present': [1 if pcc == 'present' else 0],
                'ba_present': [1 if ba == 'present' else 0],
                'bgr': [bgr]
            }
            
            # Create DataFrame and ensure all expected columns are present
            input_df = pd.DataFrame(input_data)
            
            # Add missing columns with 0
            for col in chronic_kidney_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[chronic_kidney_features]
            
            # Make prediction
            prediction = chronic_kidney_model.predict_proba(input_df)[0][1]
            risk_percentage = round(prediction * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High Risk of Chronic Kidney Disease: {risk_percentage}%")
                st.write("""
                **Recommendations:**
                - Consult a nephrologist for further evaluation
                - Monitor kidney function regularly
                - Manage blood pressure and diabetes if present
                - Follow a kidney-friendly diet
                """)
            else:
                st.success(f"Low Risk of Chronic Kidney Disease: {100 - risk_percentage}%")
                st.write("""
                **Maintain your kidney health:**
                - Stay hydrated
                - Maintain a healthy blood pressure
                - Control blood sugar levels
                - Get regular check-ups
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
