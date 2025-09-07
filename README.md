# Health Prediction System

A comprehensive web application that predicts multiple diseases using machine learning models. This application helps in early detection and risk assessment of various health conditions.

## 🌟 Features

- **Multiple Disease Prediction**: Assess risk for 6 different health conditions
  - ❤️ Heart Disease
  - 🧠 Parkinson's Disease
  - 🫁 Lung Cancer
  - 💉 Diabetes
  - 🧠 Stroke
  - 🩺 Chronic Kidney Disease
  - 🩺 Asthma
  - 🩺 CBC
- **User-friendly Interface**: Simple and intuitive web interface
- **Real-time Prediction**: Get instant risk assessment
- **Responsive Design**: Works on both desktop and mobile devices

## 🛠️ Prerequisites

- Python 3.8+
- pip (Python package installer)

## 🚀 Installation & Deployment

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-Project-for-Check-the-patient
   ```

2. **Set up the environment**
   - **Using setup script (Linux/Mac):**
     ```bash
     chmod +x setup.sh
     ./setup.sh
     ```
   - **Manual setup:**
     ```bash
     # Create and activate virtual environment
     python -m venv venv
     source venv/bin/activate  # On Windows: .\venv\Scripts\activate
     
     # Install dependencies
     pip install -r requirements.txt
     ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Cloud Deployment Options

#### 1. Streamlit Sharing (Recommended)

1. Push your code to a GitHub repository
2. Sign up at [Streamlit Sharing](https://share.streamlit.io/)
3. Click 'New App' and connect your GitHub repository
4. Select the branch and main file (app.py)
5. Click 'Deploy!'

<!--2. Render  -->
#### 2. Render

1. Create a free account at [Render](https://render.com/)
2. Install the [Render CLI](https://render.com/docs/render-cli)
3. Initialize your Render app:


<!-- #### 2. Heroku

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Login to Heroku:
   ```bash
   heroku login
   ``` -->
<!-- 3. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```
4. Deploy your code:
   ```bash
   git push heroku main
   ``` -->

#### 3. Docker (For any platform)

1. Build the Docker image:
   ```bash
   docker build -t health-prediction-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 health-prediction-app
   ```

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-Project-for-Check-the-patient
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Required Datasets

Ensure you have the following datasets in your project directory:
- `diabetes.csv`
- `heart.csv`
- `parkinsons.csv`
- `lung_cancer.csv`
- `stroke-data.csv`
- `ChronicKidneyDisease_EHRs_from_AbuDhabi.csv`
- `asthma_disease_data.csv`
-  `csb_blood_test.csv `

## 🏃‍♂️ Running the Application

1. Train the models (optional - pre-trained models are included):
   ```bash
   python train_models.py
   ```

2. Start the web application:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## 🧠 Models Used

- **Logistic Regression**: Used for most predictions
- **Random Forest**: For improved accuracy in specific conditions
- **Support Vector Machines (SVM)**: For certain classification tasks

## 📊 Data Sources

- Diabetes Dataset: Pima Indians Diabetes Database
- Heart Disease Dataset: UCI Machine Learning Repository
- Parkinson's Dataset: UCI Machine Learning Repository
- Lung Cancer Dataset: Custom dataset
- Stroke Prediction Dataset: [Source]
- Chronic Kidney Disease Dataset: EHRs from Abu Dhabi
- CSB blood test Dataset: [Source]
-Asthama diseases dataset

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Various open-source machine learning libraries
- Publicly available healthcare datasets
- Streamlit for the web interface
