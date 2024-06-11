# AI-Driven-Application-For-Diabetes-Care: Predictive-Analayis-and-Personalized-Recommendations

![image](https://github.com/ManishaLagisetty/AI-Driven-Application-For-Diabetes-Care_Predictive-Analyis-and-Personalized-Recommendations/assets/147951099/fd93fcd7-030d-4ae8-8f2b-11f9d7c8b2c7)


## Description
This project focuses on predicting diabetes using machine learning techniques. It includes data collection, preprocessing, model development, evaluation and real-time user-friendly interface.

## Abstract
This project stands out in the realm of data mining for diabetes prediction by integrating advanced sampling techniques and optimization of classification algorithms to address the critical challenge of class imbalance - a prevalent issue often overlooked in standard predictive models. Enhanced model sensitivity and precision through targeted data preprocessing and the strategic use of SMOTE, ADASYN, and RandomOverSampler with ensemble and classification classifiers. This method not only improves detection rates of diabetic cases, as evidenced by our enhanced recall and F1 scores but also reduces the risk of false positives, a crucial aspect in medical diagnostics. This comprehensive and balanced approach ensures the model's superiority in both predicting diabetes accurately and ensuring the model's applicability in real-world clinical settings.

Application also incorporates a real-time user-friendly interface allowing for immediate, personalized risk assessments in predicting diabetes and health advice that provides tailored dietary and lifestyle recommendations, enhancing user engagement and promoting proactive health management.

#### Dataset file
DiabetesDataset.csv

#### Code Implementation 
In code folder
- common_code.py --> contains data loading and preprocessing steps
- data_eda.py --> contains explonatory data analysis steps
- model_train.py --> contains model development and evaluation steps
- best_model.py --> contains the best saved model and influence of confounding variable steps
- web_app.py --> web application

#### Standardization pickle file 
scaler.pkl

#### best saved model pickle file
xgboost_ros_model.pkl

#### Technical Report
ProjectReport.pdf

#### --------Install Required Python packages--------
- pip install seaborn scikit-learn xgboost imbalanced-learn streamlit langchain
- pip install streamlit pickle-mixin python-dotenv langchain openai

#### --------To run on python virtual environment please use below commands--------
- Create a new virtual environment
- python -m venv myenv

#### --------Activate the virtual environment--------
- source myenv/bin/activate  --> For macOS/Linux
- myenv\Scripts\activate  --> For Windows

#### --------Install the necessary packages--------
- pip install langchain==0.0.316 openai==0.28.1
(Or)
- pip install langchain==0.1.17 openai==1.26.0

#### --------Run the EDA code--------
python code/data_eda.py

#### --------Run the model code--------
- python code/model_train.py
- python code/model_train.py --show-plots

#### --------Run the best model--------
- python code/best_model.py
- python code/best_model.py --show-plots

NOTE: Use --show-plots to see plots while training models or while running best model

#### --------Web Application--------
  cd code
  
  streamlit run web_app.py

