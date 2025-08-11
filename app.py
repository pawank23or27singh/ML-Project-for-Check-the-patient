import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Set page config first
st.set_page_config(page_title="Health Prediction App", page_icon="🏥", layout="wide")

# Main Application
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Home', 'Heart Disease Prediction', 'Parkinsons Prediction', 
         'Lung Cancer Prediction', 'Diabetes Prediction', 'Stroke Prediction', 
         'Chronic Kidney Disease Prediction'],
        icons=['house', 'heart', 'person', 'lungs', 'clipboard2-pulse', 'activity', 'droplet'],
        default_index=0
    )

# Home Page
if selected == 'Home':
    st.title('Multiple Disease Prediction System')
    st.write('Welcome to the Multiple Disease Prediction System. Please select a prediction model from the sidebar.')
    

    st.subheader('Available Predictions')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(''':hospital: **Heart Disease**  
        Predict your risk of heart disease''') 
        
        st.markdown(''':brain: **Parkinson's Disease**  
        Early detection of Parkinson's''') 

        st.markdown(''':hospital: **Stroke**  
        Predict your risk of stroke''')
        
    with col2:
        st.markdown(''':lungs: **Lung Cancer**  
        Assess your lung cancer risk''') 
        
        st.markdown(''':drop_of_blood: **Diabetes**  
        Check your diabetes risk''') 

        st.markdown(''':hospital: **Chronic Kidney Disease**  
        Assess your risk of chronic kidney disease''') 
    
diabetes_model = None

# Diabetes Model
if os.path.exists('diabetes.csv'):
    try:
        diabetes_data = pd.read_csv('diabetes.csv')
        if 'Outcome' not in diabetes_data.columns:
            st.sidebar.error("Error: 'Outcome' column not found in diabetes.csv")
        else:
            X = diabetes_data.drop(columns=['Outcome'])
            y = diabetes_data['Outcome']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            diabetes_model = LogisticRegression(max_iter=1000, class_weight='balanced')
            diabetes_model.fit(X_train, y_train)
            diabetes_features = X.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Error loading diabetes model: {str(e)}")
else:
    st.sidebar.error('Missing diabetes.csv file. Diabetes prediction will not work.')


if selected == 'Diabetes Prediction':
    st.title('Diabetes Risk Assessment')
    st.write('Please provide the following information for diabetes risk evaluation:')
    
    with st.form('diabetes_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0,
                                       help='Number of times pregnant')
            glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100,
                                   help='Plasma glucose concentration')
            blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=70,
                                          help='Diastolic blood pressure')
            skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20,
                                          help='Triceps skin fold thickness')
            
        with col2:
            insulin = st.number_input('Insulin Level (μU/mL)', min_value=0, max_value=1000, value=80,
                                   help='2-Hour serum insulin')
            bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                               help='Body Mass Index')
            diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, 
                                             value=0.5, step=0.01,
                                             help='Diabetes pedigree function')
            age = st.number_input('Age', min_value=1, max_value=120, value=30)
        
        submitted = st.form_submit_button('Assess Diabetes Risk', type='primary', use_container_width=True)
        
        if submitted:
            if diabetes_model is not None:
                try:
                    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                             insulin, bmi, diabetes_pedigree, age]],
                                           columns=diabetes_features)
                    
                    prediction = diabetes_model.predict(input_data)[0]
                    proba = diabetes_model.predict_proba(input_data)[0][1] * 100
                    
                  
                    st.subheader('Risk Assessment Results')
                    
                    if prediction == 1:
                        st.error(f'High Risk of Diabetes ({proba:.1f}% probability)')
                        
                        with st.expander('📌 Recommendations'):
                            st.markdown('''
                            - **Consult a healthcare professional** for further evaluation
                            - Monitor your **blood sugar levels** regularly
                            - Maintain a **healthy diet** and **exercise routine**
                            - Reduce intake of **sugary** and **processed foods**
                            - Get regular **health check-ups**
                            - Maintain a **healthy weight**
                            - Stay **physically active** (at least 150 minutes per week)
                            - Monitor your **blood pressure** and **cholesterol levels**
                            ''')
                    else:
                        st.success(f'Low Risk of Diabetes ({100 - proba:.1f}% probability)')
                        
                        with st.expander('💡 Preventive Measures'):
                            st.markdown('''
                            - Maintain a **healthy lifestyle**
                            - Exercise **regularly**
                            - Eat a **balanced diet** rich in fruits and vegetables
                            - Limit **sugar** and **processed foods**
                            - Maintain a **healthy weight**
                            - Get regular **health screenings**
                            - Stay **hydrated**
                            - Manage **stress** effectively
                            ''')
                    
                    st.markdown('---')
                    
                    st.warning('''
                    **Disclaimer:** This prediction is for informational purposes only and is not a substitute for 
                    professional medical advice. Always consult with a healthcare provider for medical advice.
                    ''')
                    
                except Exception as e:
                    st.error('❌ An error occurred during prediction. Please try again.')
                    
                    # Detailed error information in expander
                    with st.expander('🛠️ Technical Details (For Support)'):
                        st.error(f'Error: {str(e)}')
                        st.code(f'''
                        Error Type: {type(e).__name__}
                        Error Message: {str(e)}
                        ''', language='python')
            else:
                st.warning('Diabetes prediction model is not available. Please check the error message in the sidebar.')


# Load Heart Disease Model
heart_model = None
heart_features = None
if os.path.exists('heart.csv'):
    try:
        heart_data = pd.read_csv('heart.csv')
        if 'target' not in heart_data.columns:
            st.sidebar.error("Error: 'target' column not found in heart.csv")
        else:
            Xh = heart_data.drop(columns=['target'])
            yh = heart_data['target']
            Xh_train, Xh_test, yh_train, yh_test = train_test_split(
                Xh, yh, test_size=0.2, random_state=42
            )
            model = LogisticRegression(
                random_state=42, 
                max_iter=5000,
                class_weight='balanced',
                solver='lbfgs',
                tol=1e-4
            )
            heart_model = model.fit(Xh_train, yh_train)
            heart_features = Xh.columns.tolist()
    except Exception as e:
        st.sidebar.error(f"Error loading heart disease model: {str(e)}")
else:
    st.sidebar.error('Missing heart.csv file. Heart disease prediction will not work.')
# heart disease prediction       
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    st.write("Please enter the following details:")

    if heart_model is None:
        st.error("Heart disease prediction model is not available. Please check the error message in the sidebar.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age (years)', min_value=0, max_value=120, value=50, step=1)
        trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=80, max_value=200, value=120, step=1)
        restecg = st.selectbox('Resting ECG', 
                             ['Normal (0)', 'ST-T wave abnormality (1)', 'Left ventricular hypertrophy (2)'],
                             index=0)
        oldpeak = st.number_input('ST Depression (0-6.2)', min_value=0.0, max_value=6.2, value=1.0, step=0.1, format="%.1f")
        thal = st.selectbox('Thalassemia', 
                          ['Normal (0)', 'Fixed defect (1)', 'Reversible defect (2)'],
                          index=0)
    with col2:
        sex = st.radio('Sex', ['Female (0)', 'Male (1)'], index=1)
        chol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=600, value=200, step=1)
        thalach = st.number_input('Max Heart Rate (bpm)', min_value=60, max_value=220, value=150, step=1)
        slope = st.selectbox('ST Slope', 
                           ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'],
                           index=1)
    with col3:
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina (0)', 'Atypical Angina (1)', 
                          'Non-Anginal Pain (2)', 'Asymptomatic (3)'],
                         index=0)
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dL', ['No (0)', 'Yes (1)'], index=0)
        exang = st.radio('Exercise Induced Angina', ['No (0)', 'Yes (1)'], index=0)
        ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0, step=1)

    if st.button('Predict Heart Disease Risk'):
        try:
            # Prepare input data
            input_data = {
                'age': [age],
                'sex': [0 if 'Female' in sex else 1],
                'cp': [int(cp.split('(')[1].strip(')'))],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [1 if 'Yes' in fbs else 0],
                'restecg': [int(restecg.split('(')[1].strip(')'))],
                'thalach': [thalach],
                'exang': [1 if 'Yes' in exang else 0],
                'oldpeak': [oldpeak],
                'slope': [int(slope.split('(')[1].strip(')'))],
                'ca': [ca],
                'thal': [int(thal.split('(')[1].strip(')'))]
            }
            
            input_df = pd.DataFrame(input_data)
        
            for col in heart_features:
                if col not in input_df.columns:
                    input_df[col] = 0  # Add missing columns with default value 0
            
            # Reorder columns to match training data
            input_df = input_df[heart_features]
            
            #prediction
            prediction = heart_model.predict(input_df)
            proba = heart_model.predict_proba(input_df)[0][1] * 100
            
            if prediction[0] == 1:
                st.error(f'⚠️ High risk of heart disease (Probability: {proba:.1f}%)')
                st.warning('Please consult a healthcare professional for further evaluation.')
            else:
                st.success(f'✅ Low risk of heart disease (Probability: {100-proba:.1f}%)')
                st.info('Maintain a healthy lifestyle and regular check-ups.')
                
        except Exception as e:
            st.error(f'An error occurred during prediction: {str(e)}')

# Parkinson's Disease Prediction
parkinsons_model = None
parkinsons_features = None

if os.path.exists('parkinsons.csv'):
    try:
        parkinsons_data = pd.read_csv('parkinsons.csv')
        
        required_columns = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 
                           'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE', 'status']
        
        missing_columns = [col for col in required_columns if col not in parkinsons_data.columns]
        if missing_columns:
            st.sidebar.error(f"Missing required columns in parkinsons.csv: {', '.join(missing_columns)}")
        else:
            feature_cols = [col for col in parkinsons_data.columns 
                          if col != 'status' and 
                          pd.api.types.is_numeric_dtype(parkinsons_data[col])]
            
            X_parkinsons = parkinsons_data[feature_cols]
            y_parkinsons = parkinsons_data['status']
        
            parkinsons_features = feature_cols
            
            Xp_train, Xp_test, yp_train, yp_test = train_test_split(
                X_parkinsons, y_parkinsons, test_size=0.2, random_state=42
            )
            
            # Train model with class balancing
            parkinsons_model = LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            )
            parkinsons_model.fit(Xp_train, yp_train)
    except Exception as e:
        st.sidebar.error(f"Error loading Parkinson's model: {str(e)}")
else:
    st.sidebar.error("Missing 'parkinsons.csv' file. Parkinson's prediction will not work.")

if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    st.write("Please enter the following voice measurement details:")
    
    if parkinsons_model is None:
        st.error("Parkinson's prediction model is not available. Please check the error message in the sidebar.")
        st.stop()

    # Display input
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Voice Frequency Features")
        fo = st.number_input('Average vocal fundamental frequency (MDVP:Fo(Hz))', 
                           min_value=80.0, max_value=350.0, value=150.0, step=0.1, format="%.1f")
        shimmer = st.number_input('Average variation in amplitude (MDVP:Shimmer(dB))', 
                                min_value=0.0, max_value=0.2, value=0.05, step=0.001, format="%.3f")
        rpde = st.number_input('Nonlinear measure of fundamental frequency (RPDE)', 
                              min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
        spread1 = st.number_input('Nonlinear measure of fundamental frequency variation (spread1)', 
                                 min_value=-10.0, max_value=0.0, value=-5.0, step=0.1, format="%.1f")

    with col2:
        st.markdown("### Voice Quality Features")
        jitter = st.number_input('Jitter percentage (MDVP:Jitter(%))', 
                               min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")
        nhr = st.number_input('Noise-to-harmonics ratio (NHR)', 
                             min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.3f")
        dfa = st.number_input('Signal fractal scaling exponent (DFA)', 
                             min_value=0.5, max_value=1.0, value=0.7, step=0.001, format="%.3f")
        spread2 = st.number_input('Nonlinear measure of fundamental frequency variation (spread2)', 
                                 min_value=0.0, max_value=0.5, value=0.2, step=0.01, format="%.2f")

    with col3:
        st.markdown("### Additional Voice Features")
        hnr = st.number_input('Harmonics-to-noise ratio (HNR)', 
                             min_value=0.0, max_value=40.0, value=20.0, step=0.1, format="%.1f")
        d2 = st.number_input('Correlation dimension (D2)', 
                            min_value=1.0, max_value=3.0, value=2.0, step=0.01, format="%.2f")
        ppe = st.number_input('Pitch period entropy (PPE)', 
                             min_value=0.0, max_value=0.5, value=0.1, step=0.001, format="%.3f")

    if st.button("Predict Parkinson's Disease Risk"):
        try:
            input_data = {
                'MDVP:Fo(Hz)': [fo],
                'MDVP:Jitter(%)': [jitter],
                'MDVP:Shimmer': [shimmer],
                'NHR': [nhr],
                'HNR': [hnr],
                'RPDE': [rpde],
                'DFA': [dfa],
                'spread1': [spread1],
                'spread2': [spread2],
                'D2': [d2],
                'PPE': [ppe]
            }
            
            input_df = pd.DataFrame(input_data)
        
            for col in parkinsons_features:
                if col not in input_df.columns:
                    input_df[col] = 0 
        
            input_df = input_df[parkinsons_features]
            
            # Make prediction
            prediction = parkinsons_model.predict(input_df)
            proba = parkinsons_model.predict_proba(input_df)[0][1] * 100
            
            if prediction[0] == 1:
                st.error(f'⚠️ High probability of Parkinson\'s disease ({proba:.1f}%)')
                st.warning('Please consult a neurologist for further evaluation.')
            else:
                st.success(f'✅ Low probability of Parkinson\'s disease ({(100-proba):.1f}%)')
                st.info('No signs of Parkinson\'s disease detected in the voice analysis.')
                
        except Exception as e:
            st.error(f'An error occurred during prediction: {str(e)}')

# Lung Cancer Prediction
lung_cancer_model = None
lung_cancer_features = None

lung_cancer_file = 'lung_cancer.csv'
if os.path.exists(lung_cancer_file):
    try:
        st.sidebar.write("\n🔍 Loading lung cancer data...")
        
        lung_cancer_data = pd.read_csv(lung_cancer_file)
        st.sidebar.write(f"✅ Loaded {len(lung_cancer_data)} rows from {lung_cancer_file}")
        
        st.sidebar.write("\n📋 First few rows of data:")
        st.sidebar.dataframe(lung_cancer_data.head(2))
        
    
        required_columns = ['Age', 'Smokes', 'AreaQ', 'Alkhol', 'Result']
        
        actual_columns = lung_cancer_data.columns.tolist()
        missing_columns = [col for col in required_columns if col not in actual_columns]
        
        if missing_columns:
            st.sidebar.error(f"❌ Missing required columns in {lung_cancer_file}:")
            st.sidebar.write("Missing:", ", ".join(missing_columns))
            st.sidebar.write("\nAvailable columns:", ", ".join(actual_columns))
        else:
            st.sidebar.success("✅ All required columns found")
            
            cols_to_drop = [col for col in ['Name', 'Surname', 'GENDER'] if col in lung_cancer_data.columns]
            if cols_to_drop:
                lung_cancer_data = lung_cancer_data.drop(columns=cols_to_drop)
                st.sidebar.write(f"Dropped columns: {', '.join(cols_to_drop)}")
            
           
            for col in lung_cancer_data.columns:
                if col != 'Result':
                    lung_cancer_data[col] = pd.to_numeric(lung_cancer_data[col], errors='coerce')
            
            lung_cancer_data['Result'] = lung_cancer_data['Result'].apply(
                lambda x: 1 if str(x).lower() in ['1', 'true', 'yes'] else 0
            )
            
            missing_values = lung_cancer_data.isnull().sum().sum()
            if missing_values > 0:
                st.sidebar.warning(f"⚠️ Found {missing_values} missing values. Rows with missing values will be dropped.")
            
            initial_rows = len(lung_cancer_data)
            lung_cancer_data = lung_cancer_data.dropna()
            rows_after_drop = len(lung_cancer_data)
            
            if rows_after_drop < initial_rows:
                st.sidebar.warning(f"Dropped {initial_rows - rows_after_drop} rows with missing values")
            
            if lung_cancer_data.empty:
                st.sidebar.error('❌ No valid data remaining after cleaning. Please check your lung_cancer.csv file.')
            else:
                st.sidebar.success(f"✅ Using {rows_after_drop} clean samples")
                
                X = lung_cancer_data.drop(columns=['Result'])
                y = lung_cancer_data['Result']
                
                lung_cancer_features = X.columns.tolist()
                st.sidebar.write("\n🔢 Features to be used:", ", ".join(lung_cancer_features))
                
                class_distribution = y.value_counts()
                st.sidebar.write("\n📊 Class distribution:", class_distribution.to_dict())
                
                if len(class_distribution) < 2:
                    st.sidebar.error("❌ Need at least 2 classes for classification")
                else:
                    # Split and train model
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    st.sidebar.write(f"\n🔧 Training model with {len(X_train)} samples...")
                    
                    try:
                        lung_cancer_model = LogisticRegression(
                            max_iter=2000, 
                            class_weight='balanced',
                            random_state=42
                        )
                        lung_cancer_model.fit(X_train, y_train)
                        
                        # Check model performance
                        train_score = lung_cancer_model.score(X_train, y_train)
                        test_score = lung_cancer_model.score(X_test, y_test)
                        
                        st.sidebar.success(f"✅ Model trained successfully!")
                        st.sidebar.write(f"Training accuracy: {train_score:.2f}")
                        st.sidebar.write(f"Test accuracy: {test_score:.2f}")
                        
                    except Exception as model_error:
                        st.sidebar.error(f"❌ Error during model training: {str(model_error)}")
                        st.sidebar.exception(model_error)
    
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {lung_cancer_file}: {str(e)}")
        st.sidebar.exception(e)
else:
    st.sidebar.error(f"❌ {lung_cancer_file} not found in the current directory.")
    st.sidebar.write("Current working directory:", os.getcwd())


# Stroke Prediction
stroke_model = None
stroke_feature_columns = None

chronic_kidney_model = None
ckd_feature_columns = None

if os.path.exists('ChronicKidneyDisease_EHRs_from_AbuDhabi.csv'):
    try:

        ckd_data = pd.read_csv('ChronicKidneyDisease_EHRs_from_AbuDhabi.csv')
        if 'EventCKD35' not in ckd_data.columns:
            st.sidebar.error("Error: 'EventCKD35' column not found in ChronicKidneyDisease_EHRs_from_AbuDhabi.csv")
        else:
            for col in ckd_data.columns:
                if col != 'EventCKD35': 
                    ckd_data[col] = pd.to_numeric(ckd_data[col], errors='coerce')
            ckd_data = ckd_data.dropna()
            
            if ckd_data.empty:
                st.sidebar.error('No valid data remaining after cleaning. Please check your Chronic Kidney Disease data file.')
            else:
                X = ckd_data.drop(columns=['EventCKD35'])
                y = ckd_data['EventCKD35']
                
                ckd_feature_columns = X.columns.tolist()
                
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
                st.sidebar.success(f"Chronic Kidney Disease model trained with {len(X_train)} samples")
                st.sidebar.write(f"Features used: {', '.join(ckd_feature_columns[:5])}...")
    except Exception as e:
        st.sidebar.error(f'Error loading Chronic Kidney Disease model: {str(e)}')
else:
    st.sidebar.error("Missing 'ChronicKidneyDisease_EHRs_from_AbuDhabi.csv' file. CKD prediction will not work.")
if os.path.exists('stroke-data.csv'):
    try:
        stroke_data = pd.read_csv('stroke-data.csv')
        
        required_columns = ['age', 'hypertension', 'heart_disease', 'ever_married', 
                          'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 
                          'smoking_status', 'stroke']
        missing_columns = [col for col in required_columns if col not in stroke_data.columns]
        
        if missing_columns:
            st.sidebar.error(f"Missing required columns in stroke-data.csv: {', '.join(missing_columns)}")
        else:
          
            stroke_data['bmi'] = pd.to_numeric(stroke_data['bmi'], errors='coerce')
            stroke_data = stroke_data.fillna({'bmi': stroke_data['bmi'].median()})
            
            stroke_data['hypertension'] = stroke_data['hypertension'].astype(int)
            stroke_data['heart_disease'] = stroke_data['heart_disease'].astype(int)
            stroke_data['ever_married'] = stroke_data['ever_married'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
            
            categorical_cols = ['work_type', 'Residence_type', 'smoking_status']
            stroke_data_encoded = pd.get_dummies(stroke_data, columns=categorical_cols, drop_first=True)

            X = stroke_data_encoded.drop(columns=['stroke'])
            y = stroke_data_encoded['stroke']
            stroke_feature_columns = X.columns.tolist()
            
            # Split and train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            stroke_model = LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            )
            stroke_model.fit(X_train, y_train)
            
    except Exception as e:
        st.sidebar.error(f'Error loading stroke prediction model: {str(e)}')
else:
    st.sidebar.error("Missing 'stroke-data.csv' file. Stroke prediction will not work.")

if selected == "Lung Cancer Prediction":
    st.title("Lung Cancer Risk Assessment")
    st.write("Please provide the following information for lung cancer risk evaluation:")
    
    if lung_cancer_model is None:
        st.error("⚠️ Lung cancer prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    
    with st.form("lung_cancer_form"):
        st.write("### Please provide the following information for lung cancer risk assessment:")

        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age (years)', min_value=0, max_value=120, value=50, step=1,
                                help="Enter your current age")
        
    
        with col2:
            smokes = st.number_input('Smoking Score (0-10)', min_value=0, max_value=10, value=0, step=1,
                                   help="Rate your smoking habits from 0 (non-smoker) to 10 (heavy smoker)")
        
      
        col1, col2 = st.columns(2)
        with col1:
            areaq = st.number_input('Air Quality Index (1-10)', min_value=1, max_value=10, value=5, step=1,
                                   help="Rate your local air quality from 1 (excellent) to 10 (very poor)")
        
        with col2:
            alkhol = st.number_input('Alcohol Consumption (0-10)', min_value=0, max_value=10, value=0, step=1,
                                    help="Rate your alcohol consumption from 0 (none) to 10 (heavy)")
        submitted = st.form_submit_button("Assess Lung Cancer Risk", type="primary", use_container_width=True)
        
        if submitted:
            with st.spinner('Analyzing your information...'):
                try:

                    input_data = {
                        'Age': [age],
                        'Smokes': [smokes],
                        'AreaQ': [areaq],
                        'Alkhol': [alkhol]
                    }
                    
                    input_df = pd.DataFrame(columns=lung_cancer_features)
                    for feature in lung_cancer_features:
                        matching_input = next((k for k in input_data.keys() if k.lower() == feature.lower()), None)
                        if matching_input:
                            input_df[feature] = input_data[matching_input]
                        else:
                            input_df[feature] = 0
                    
                    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    for col in lung_cancer_features:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[lung_cancer_features]
                    
                    # Make prediction
                    prediction = lung_cancer_model.predict(input_df)
                    proba = lung_cancer_model.predict_proba(input_df)[0][1] * 100
            
                    st.markdown("## 🩺 Lung Cancer Risk Assessment Results")
                    if prediction[0] == 1 or proba >= 50:
                        risk_level = "High"
                        risk_color = "#ff4b4b"  # Red
                        icon = "⚠️"
                        recommendations = [
                            "Consult with a healthcare professional for further evaluation.",
                            "If you smoke, seek help to quit immediately.",
                            "Avoid exposure to secondhand smoke and other lung irritants.",
                            "Consider discussing screening options with your doctor."
                        ]
                    else:
                        risk_level = "Low"
                        risk_color = "#4CAF50" 
                        icon = "✅"
                        recommendations = [
                            "Maintain a healthy lifestyle with regular exercise.",
                            "If you smoke, consider quitting to reduce future risks.",
                            "Get regular check-ups and report any respiratory symptoms.",
                            "Maintain good indoor air quality in your home."
                        ]
                    st.markdown(f"""
                    <div style='background-color:#f8f9fa; padding:20px; border-radius:10px; margin:10px 0;'>
                        <h2 style='color:{risk_color};'>{icon} {risk_level} Risk of Lung Cancer</h2>
                        <p>Based on the provided information, your estimated risk is <strong>{proba:.1f}%</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### Recommendations:")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    with st.expander("ℹ️ More about lung cancer risk factors"):
                        st.markdown("""
                        **Factors that may increase lung cancer risk include:**
                        - Smoking (current or former)
                        - Exposure to secondhand smoke
                        - Exposure to radon gas
                        - Occupational exposure to carcinogens
                        - Family history of lung cancer
                        - Air pollution
                        """)
                    with st.expander("ℹ️ Technical Details"):
                        st.write("### Model Information")
                        st.write(f"Model type: {type(lung_cancer_model).__name__}")
                        st.write(f"Features used: {', '.join(lung_cancer_features[:5])}...")
                        st.write(f"Prediction confidence: {max(proba, 100-proba):.1f}%")
                        
                        st.write("### Input Summary")
                        st.json({
                            "Age": age,
                            "Smoking Status": smokes,
                            "AreaQ": areaq,
                            "Alkhol": alkhol
                        })
                    
                    # Prevention tips
                    st.info('### Prevention Tips:')
                    st.markdown("""
                    - Avoid all forms of tobacco and secondhand smoke
                    - Test your home for radon and take steps to reduce levels if high
                    - Follow workplace safety guidelines if exposed to carcinogens
                    - Eat a diet rich in fruits and vegetables
                    - Exercise most days of the week
                    - Get regular medical check-ups
                    """)
                
                except Exception as e:
                    st.error("❌ An error occurred during prediction. Please try again or contact support if the issue persists.")
                    with st.expander("🛠️ Technical Details (For Support)"):
                        st.write("### Error Information")
                        st.write(f"Error type: `{type(e).__name__}`")
                        st.write(f"Error message: `{str(e)}`")
                        
                        st.write("### Debug Information")
                        try:
                            st.write("#### Input Data")
                            st.json(input_data)
                            
                            st.write("#### Available Features")
                            st.write(lung_cancer_features)
                            
                            st.write("#### Model Information")
                            if lung_cancer_model is not None:
                                st.write(f"Model type: {type(lung_cancer_model).__name__}")
                                st.write(f"Model parameters: {lung_cancer_model.get_params()}")
                            else:
                                st.warning("Model not initialized")
                                
                        except Exception as debug_e:
                            st.error(f"Error generating debug information: {str(debug_e)}")
                    

                    st.markdown("""
                    **What to do next:**
                    1. Check if all required fields are filled correctly
                    2. Try refreshing the page and submitting again
                    3. If the problem persists, please contact support with the error details above
                    """)
                
                    st.info('### Prevention Tips:')
                    st.markdown("""
                    - Avoid smoking and secondhand smoke
                    - Maintain a healthy diet and exercise routine
                    - Limit alcohol consumption
                    - Get regular check-ups with your healthcare provider
                    """)
                
                    # Disclaimer
                    st.markdown("---")
                    st.caption("""
                    **Note:** This assessment is for informational purposes only and is not a substitute for 
                    professional medical advice, diagnosis, or treatment. Always seek the advice of your 
                    physician or other qualified health provider with any questions you may have 
                    regarding a medical condition.
                    """)
# Stroke Prediction UI
if selected == "Stroke Prediction":
    st.title("Stroke Risk Assessment")
    st.write("Please provide the following information to assess your stroke risk:")
    
    if stroke_model is None:
        st.error("Stroke prediction model is not available. Please check the data files.")
    else:
        with st.form("stroke_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
                hypertension = st.radio("Do you have hypertension?", ["No", "Yes"])
                heart_disease = st.radio("Do you have heart disease?", ["No", "Yes"])
                avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
                
            with col2:
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
                ever_married = st.radio("Have you ever been married?", ["No", "Yes"])
                work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
                residence_type = st.radio("Residence Type", ["Urban", "Rural"])
                smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
            
            submitted = st.form_submit_button("Assess Stroke Risk")
            
            if submitted:
                try:
                    
                    input_data = {
                        'age': [age],
                        'hypertension': [1 if hypertension == "Yes" else 0],
                        'heart_disease': [1 if heart_disease == "Yes" else 0],
                        'ever_married': [1 if ever_married == "Yes" else 0],
                        'avg_glucose_level': [avg_glucose_level],
                        'bmi': [bmi],
                        'work_type_' + work_type: [1],
                        'Residence_type_' + residence_type: [1],
                        'smoking_status_' + smoking_status.replace(" ", "_"): [1]
                    }
                    
                    input_df = pd.DataFrame(0, index=[0], columns=stroke_feature_columns)
                    
                    for col, value in input_data.items():
                        if col in input_df.columns:
                            input_df[col] = value
                    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    # Make prediction
                    prediction = stroke_model.predict(input_df)
                    proba = stroke_model.predict_proba(input_df)[0][1] * 100
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Assessment Results")
                    
                    if prediction[0] == 1:
                        st.error(f"🚨 High Risk of Stroke ({proba:.1f}% probability)")
                        st.warning("""
                        **Please consult a healthcare professional immediately.**
                        
                        Based on the information provided, you may be at an elevated risk of stroke. 
                        Consider these immediate actions:
                        - Contact your healthcare provider
                        - Monitor your blood pressure regularly
                        - Review your lifestyle choices
                        """)
                    else:
                        stroke_free_prob = 100 - proba
                        if stroke_free_prob > 90:
                            risk_level = "Very Low"
                            risk_color = "#0068c9" 
                        elif stroke_free_prob > 70:
                            risk_level = "Low"
                            risk_color = "#00cc96"  
                        elif stroke_free_prob > 50:
                            risk_level = "Moderate"
                            risk_color = "#ffcc00"
                        else:
                            risk_level = "High"
                            risk_color = "#ff7f0e"  
                        
                        st.success(f"✅ {risk_level} Risk of Stroke ({stroke_free_prob:.1f}% probability of being stroke-free)")
                        
                        # Add prevention tips
                        st.info("### Prevention Tips")
                        st.markdown("""
                        - Maintain a healthy blood pressure (below 120/80 mmHg)
                        - Control cholesterol and blood sugar levels
                        - Exercise regularly (at least 150 minutes per week)
                        - Eat a balanced diet rich in fruits and vegetables
                        - Limit alcohol consumption
                        - Avoid smoking and secondhand smoke
                        - Manage stress through relaxation techniques
                        """)
                    
                    # Add disclaimer
                    st.markdown("---")
                    st.caption("""
                    **Note:** This assessment is for informational purposes only and is not a substitute for 
                    professional medical advice, diagnosis, or treatment. Always seek the advice of your 
                    physician or other qualified health provider with any questions you may have 
                    regarding a medical condition.
                    """)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    with st.expander("Technical Details"):
                        st.write(f"Error type: {type(e).__name__}")
                        st.write(f"Error message: {str(e)}")

# Chronic Kidney Disease Prediction
chronic_kidney_disease_model = None
chronic_kidney_disease_features=None

if os.path.exists('ChronicKidneyDisease_EHRs_from_AbuDhabi.csv'):
    try:
       
        ckd_data = pd.read_csv('ChronicKidneyDisease_EHRs_from_AbuDhabi.csv')
        
        if 'EventCKD35' not in ckd_data.columns:
            st.sidebar.error("Required column 'EventCKD35' not found in the dataset.")
        else:
            numeric_cols = ckd_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col != 'EventCKD35': 
                    ckd_data[col] = pd.to_numeric(ckd_data[col], errors='coerce')
            median_val = ckd_data[col].median()
            ckd_data = ckd_data.fillna({col: median_val})
            
            X = ckd_data.drop(columns=['EventCKD35'])
            y = ckd_data['EventCKD35']
            
            ckd_feature_columns = X.columns.tolist()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            chronic_kidney_disease_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            chronic_kidney_disease_model.fit(X_train, y_train)
            
    except Exception as e:
        st.sidebar.error(f'Error loading CKD model: {str(e)}')
else:
    st.sidebar.error("Missing 'ChronicKidneyDisease_EHRs_from_AbuDhabi.csv' file. CKD prediction will not work.")

if selected == "Chronic Kidney Disease Prediction":
    st.title("Chronic Kidney Disease Risk Assessment")
    st.write("Please provide the following medical information:")
    
    if chronic_kidney_model is None:
        st.error("Chronic Kidney Disease prediction model is not available. Please check the error message in the sidebar.")
        st.stop()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        age = st.number_input('Age (years)', min_value=0, max_value=120, value=50, step=1,
                            help="Enter your current age")
        
        sex = st.radio('Sex', ['Male', 'Female'], 
                      help="Biological sex at birth")
        
        st.markdown("### Medical History")
        htn = st.radio('Hypertension (HTN)', ['No', 'Yes'],
                      help="Have you been diagnosed with high blood pressure?")
        
        dm = st.radio('Diabetes Mellitus (DM)', ['No', 'Yes'],
                     help="Have you been diagnosed with diabetes?")
        
        cad = st.radio('Coronary Artery Disease (CAD)', ['No', 'Yes'],
                      help="Have you been diagnosed with coronary artery disease?")
        
        st.markdown("### Laboratory Results")
        cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=0, max_value=500, value=180, step=1,
                                    help="Your most recent cholesterol level")
        
        sg = st.selectbox('Specific Gravity (SG)', ['Low', 'High'],
                         help="Urine specific gravity test result")
        
        al = st.selectbox('Albumin (AL)', ['Low', 'High'],
                         help="Albumin level in urine")
        
        su = st.selectbox('Sugar (SU)', ['Low', 'High'],
                         help="Sugar level in urine")
        
    with col2:
        st.markdown("### Lifestyle Factors")
        smoking_status = st.selectbox('Smoking Status', 
                                    ['never smoked', 'formerly smoked', 'smokes'],
                                    help="Your current smoking status")
        
        alcohol_intake = st.number_input('Alcohol Intake (drinks/week)', min_value=0, max_value=100, value=0, step=1,
                                       help="Average number of alcoholic drinks per week")
        
        ever_married = st.radio('Ever Married', ['No', 'Yes'],
                               help="Have you ever been married?")
        
        work_type = st.selectbox('Work Type', 
                               ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
                               help="Your current or most recent work type")
        
        residence_type = st.radio('Residence Type', ['Urban', 'Rural'],
                                 help="Do you live in an urban or rural area?")
        
        st.markdown("### Additional Lab Results")
        bp = st.selectbox('Blood Pressure (BP)', ['Normal', 'High'],
                         help="Your blood pressure status")
        
        rbc = st.selectbox('Red Blood Cells (RBC)', ['Normal', 'Abnormal'],
                          help="Red blood cell count")
        
        pc = st.selectbox('Pus Cell (PC)', ['Normal', 'Abnormal'],
                         help="Pus cell presence in urine")
        
        pcc = st.selectbox('Pus Cell Clumps (PCC)', ['Not Present', 'Present'],
                          help="Pus cell clumps in urine")
        
        ba = st.selectbox('Bacteria (BA)', ['Not Present', 'Present'],
                         help="Bacteria presence in urine")
        
        appet = st.selectbox('Appetite', ['Good', 'Poor'],
                           help="Your general appetite")
        
        pe = st.selectbox('Pedal Edema (PE)', ['No', 'Yes'],
                         help="Swelling in the lower extremities")
        
        ane = st.selectbox('Anemia (ANE)', ['No', 'Yes'],
                          help="Have you been diagnosed with anemia?")
    st.markdown("---")
    if st.button("Assess CKD Risk", type="primary", use_container_width=True):
        try:
            input_data = {
                'age': [age],
                'sex': [1 if sex == 'Male' else 0],
                'hypertension': [1 if htn == 'Yes' else 0],
                'diabetes': [1 if dm == 'Yes' else 0],
                'cholesterol': [cholesterol],
                'specific_gravity': [1 if sg == 'High' else 0],
                'albumin': [1 if al == 'High' else 0],
                'sugar': [1 if su == 'High' else 0]
            }
            additional_features = {
                'smoking_status': [0],
                'alcohol_intake': [0],
                'ever_married': [0],
                'work_type': [0],
                'residence_type': [0],
                'bp': [0],
                'rbc': [0],
                'pc': [0],
                'pcc': [0],
                'ba': [0],
                'cad': [1 if cad == 'Yes' else 0],
                'appet': [0],
                'pe': [0],
                'ane': [0]
            }
            input_data.update(additional_features)
            input_df = pd.DataFrame(input_data)
            for col in ckd_feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0 
            input_df = input_df[ckd_feature_columns]
            
            # Make prediction
            prediction = chronic_kidney_model.predict(input_df)
            proba = chronic_kidney_model.predict_proba(input_df)[0][1] * 100
            st.markdown("## Risk Assessment Results")
            
            if prediction[0] == 1:
                st.error(f'⚠️ High risk of Chronic Kidney Disease ({proba:.1f}%)')
                st.warning('### Recommendations:')
                st.markdown("""
                - Consult with a nephrologist (kidney specialist) for further evaluation
                - Monitor and control your blood pressure and blood sugar levels
                - Follow a kidney-friendly diet (low sodium, controlled protein)
                - Stay hydrated and avoid NSAID pain relievers
                - Manage underlying conditions (diabetes, hypertension)
                - Get regular kidney function tests
                """)
            else:
                st.success(f'✅ Low risk of Chronic Kidney Disease ({(100-proba):.1f}%)')
                st.info('### Prevention Tips:')
                st.markdown("""
                - Maintain a healthy blood pressure (below 130/80 mmHg)
                - Control blood sugar levels if you have diabetes
                - Stay hydrated by drinking plenty of water
                - Eat a balanced diet low in salt and processed foods
                - Exercise regularly and maintain a healthy weight
                - Avoid smoking and limit alcohol consumption
                - Get regular check-ups if you have risk factors
                """)
            st.markdown("---")
            st.caption("""
            **Note:** This assessment is for informational purposes only and is not a substitute for 
            professional medical advice, diagnosis, or treatment. Always seek the advice of your 
            physician or other qualified health provider with any questions you may have 
            regarding a medical condition.
            """)
                
        except Exception as e:
            st.error(f'An error occurred during prediction: {str(e)}')
            with st.expander("Technical Details"):
                st.write("### Debug Information")
                st.write(f"Error type: {type(e).__name__}")
                st.write(f"Error message: {str(e)}")
                if 'input_encoded' in locals():
                    st.write("### Input Data")
                    st.write(input_encoded)
                if ckd_feature_columns is not None:
                    st.write("### Expected Features")
                    st.write(ckd_feature_columns)
