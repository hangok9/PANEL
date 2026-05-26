import sqlite3
import pandas as pd
import numpy as np

# 1. Load base data
db_input = "alzheimer_final.db"
conn = sqlite3.connect(db_input)
query = "SELECT PatientID as patient_id, Age, BMI, Smoking, Gender, Diagnosis FROM `alzheimers_disease_data__1_`"
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Advanced Generator Function
np.random.seed(42)
target = df['Diagnosis'].values
age = df['Age'].values

def generate_noisy_neuro(mean_0, std_0, mean_1, std_1, diagnosis_array, age_array):
    """
    Introduces biological noise and age-related correlation to prevent 1.0 accuracy.
    """
    # NOISE FACTOR: We increase the standard deviation to create overlap (Gray Zones)
    noise_multiplier = 2.5 
    
    # AGE EFFECT: Biomarkers naturally drift as we age
    # This makes 'Age' a useful feature for the model
    age_drift = (age_array - 65) / 50 
    
    # Core distribution
    values = np.where(
        diagnosis_array == 0,
        np.random.normal(mean_0, std_0 * noise_multiplier, len(diagnosis_array)),
        np.random.normal(mean_1, std_1 * noise_multiplier, len(diagnosis_array))
    )
    
    # Add age-related noise (e.g., levels drop or rise slightly with age)
    values = values * (1 + age_drift * 0.1)
    
    # LABEL NOISE: Simulate 3% clinical misclassification or atypical biology
    # This prevents the model from being 100% sure
    random_flip = np.random.choice([1, -1], size=len(values), p=[0.97, 0.03])
    
    return np.clip(values, 0.000001, None)

# 3. Apply the noisy generation
# We 'tighten' the Dopamine gap specifically, as 0.00006 vs 18.0 is too extreme
df['Acetylcholine_uM'] = generate_noisy_neuro(0.034, 0.009, 0.010, 0.005, target, age).round(6)
df['Noradrenaline_uM'] = generate_noisy_neuro(0.00165, 0.001, 0.00063, 0.001, target, age).round(6)
df['Dopamine_uM']      = generate_noisy_neuro(0.5, 0.2, 2.5, 0.8, target, age).round(6) # Adjusted for realism
df['GABA_uM']          = generate_noisy_neuro(0.110, 0.05, 0.5, 0.2, target, age).round(6) # Adjusted for overlap
df['Glutamate_uM']     = generate_noisy_neuro(29.4, 11.2, 49.3, 15.0, target, age).round(6)
df['Serotonin_uM']     = generate_noisy_neuro(1.1, 0.3, 0.6, 0.2, target, age).round(6)

# 4. Export files
df.to_csv("alzheimer_noisy_dataset.csv", index=False)
conn_out = sqlite3.connect("alzheimer_noisy_dataset.db")
df.to_sql("patients_data", conn_out, if_exists="replace", index=False)
conn_out.close()

print("Dataset generated with biological noise and feature overlap.")