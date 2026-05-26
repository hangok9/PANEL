import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn & XGBoost Imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score

# Set plot style for professional clinical reporting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def load_and_preprocess_data(filepath):
    """Loads the dataset and establishes the preprocessing pipeline."""
    df = pd.read_csv(filepath)
    
    target = 'Diagnosis'
    numerical_features = [
        'Age', 'BMI', 'Acetylcholine_uM', 'Noradrenaline_uM', 
        'Dopamine_uM', 'GABA_uM', 'Glutamate_uM', 'Serotonin_uM'
    ]
    categorical_features = ['Smoking', 'Gender']
    
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Trains, tunes, and evaluates RF and XGBoost models."""
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])
    
    rf_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
    
    xgb_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0]
    }

    print("\n[*] Initiating Hyperparameter Tuning (Optimizing for F1-Score)...")
    
    rf_search = RandomizedSearchCV(rf_pipeline, param_distributions=rf_param_grid, 
                                   n_iter=10, scoring='f1', cv=5, random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(xgb_pipeline, param_distributions=xgb_param_grid, 
                                    n_iter=10, scoring='f1', cv=5, random_state=42, n_jobs=-1)

    rf_search.fit(X_train, y_train)
    xgb_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_
    best_xgb = xgb_search.best_estimator_

    models = {'Random Forest': best_rf, 'XGBoost': best_xgb}
    best_model_name = None
    best_f1 = 0
    best_model = None

    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n=> Champion Model Selected: {best_model_name} (F1-Score: {best_f1:.4f})")
    return best_model, best_model_name

def plot_evaluation_metrics(model, model_name, X_test, y_test):
    """Generates the Confusion Matrix, ROC-AUC, and Feature Importance plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'Confusion Matrix ({model_name})')
    axes[0].set_xticks([0.5, 1.5], ['Healthy (0)', 'Alzheimer\'s (1)'])
    axes[0].set_yticks([0.5, 1.5], ['Healthy (0)', 'Alzheimer\'s (1)'])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    preprocessor = model.named_steps['preprocessor']
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out().tolist()
    feature_names = num_features + cat_features

    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    sns.barplot(x=sorted_importances, y=sorted_features, ax=axes[2], hue=sorted_features, legend=False, palette='viridis')
    axes[2].set_title('Feature Importance')

    plt.tight_layout()
    
    # Alert the user that the program is paused!
    print("\n" + "!"*60)
    print("! ACTION REQUIRED: Close the plot window to continue... !")
    print("!"*60 + "\n")
    
    plt.show() # Program pauses here until closed

def get_patient_data_from_user():
    """Prompts the user in the terminal to input patient data one by one."""
    print("\n" + "="*50)
    print("      ENTER NEW PATIENT BIOMARKERS")
    print("="*50)
    
    patient_dict = {}
    
    def get_numeric_input(prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("  [!] Invalid input. Please enter a number (e.g., 0.05).")

    def get_category_input(prompt, valid_options):
        while True:
            try:
                val = int(input(prompt))
                if val in valid_options:
                    return val
                else:
                    print(f"  [!] Please enter one of the valid options: {valid_options}")
            except ValueError:
                print("  [!] Invalid input. Please enter a whole number.")

    patient_dict['Age'] = get_numeric_input("Age (years)               : ")
    patient_dict['BMI'] = get_numeric_input("BMI                       : ")
    
    print("\n--- Neurotransmitter Levels (µM) ---")
    patient_dict['Acetylcholine_uM'] = get_numeric_input("Acetylcholine (e.g., 0.04) : ")
    patient_dict['Noradrenaline_uM'] = get_numeric_input("Noradrenaline (e.g., 0.15) : ")
    patient_dict['Dopamine_uM']      = get_numeric_input("Dopamine (e.g., 0.05)      : ")
    patient_dict['GABA_uM']          = get_numeric_input("GABA (e.g., 0.10)          : ")
    patient_dict['Glutamate_uM']     = get_numeric_input("Glutamate (e.g., 18.5)     : ")
    patient_dict['Serotonin_uM']     = get_numeric_input("Serotonin (e.g., 0.06)     : ")
    
    print("\n--- Categorical Data ---")
    patient_dict['Smoking'] = get_category_input("Smoking (0 = No, 1 = Yes)  : ", [0, 1])
    patient_dict['Gender']  = get_category_input("Gender (0 = Female, 1 = Male): ", [0, 1])
    
    return patient_dict

def predict_new_patient(model, patient_dict):
    """Calculates and prints the probability of an Alzheimer's diagnosis."""
    patient_df = pd.DataFrame([patient_dict])
    
    probabilities = model.predict_proba(patient_df)[0]
    prob_healthy = probabilities[0] * 100
    prob_alzheimer = probabilities[1] * 100
    
    print("\n" + "="*50)
    print("      NEW PATIENT DIAGNOSTIC PREDICTION ")
    print("="*50)
    print(f"Probability of Healthy (0):     {prob_healthy:.2f}%")
    print(f"Probability of Alzheimer's (1): {prob_alzheimer:.2f}%")
    
    if prob_alzheimer > 50.0:
        print("\n=> MODEL CLASSIFICATION: High Risk for Alzheimer's Disease")
    else:
        print("\n=> MODEL CLASSIFICATION: Low Risk (Healthy Control)")
    print("="*50 + "\n")

if __name__ == "__main__":
    DATA_FILEPATH = 'alzheimer_noisy_dataset.csv'
    MODEL_FILEPATH = 'alzheimers_champion_model.joblib'
    
    # 1. Model Loading / Training Phase
    if os.path.exists(MODEL_FILEPATH):
        print(f"[*] Found existing model at '{MODEL_FILEPATH}'. Loading it now...")
        champion_model = joblib.load(MODEL_FILEPATH)
        print("[+] Model loaded successfully.")
    else:
        print(f"[*] No saved model found. Training a new one from '{DATA_FILEPATH}'...")
        try:
            X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(DATA_FILEPATH)
            champion_model, champion_name = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
            
            # Will pause here until window is closed!
            plot_evaluation_metrics(champion_model, champion_name, X_test, y_test)
            
            joblib.dump(champion_model, MODEL_FILEPATH)
            print(f"[+] Model successfully saved as '{MODEL_FILEPATH}' for future use.")
        except FileNotFoundError:
            print(f"Error: Could not locate '{DATA_FILEPATH}'.")
            exit()

    # 2. Interactive Terminal Loop
    while True:
        try:
            new_patient = get_patient_data_from_user()
            print("\nProcessing patient data...")
            predict_new_patient(champion_model, new_patient)
            
            run_again = input("Would you like to analyze another patient? (y/n): ").strip().lower()
            if run_again != 'y':
                print("Exiting diagnostic tool. Goodbye!")
                break
        except KeyboardInterrupt:
            # Lets you exit gracefully if you press Ctrl+C
            print("\nExiting diagnostic tool. Goodbye!")
            break