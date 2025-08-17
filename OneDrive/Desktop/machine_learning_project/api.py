from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime

# Load model and encoders
model = joblib.load("model.joblib")  # Update path to your model file
label_encoders = joblib.load("label_encoders.joblib")  # Update path to your encoders

app = FastAPI()

class InsuranceData(BaseModel):
    Age: float
    Gender: str
    Annual_Income: float
    Marital_Status: str
    Number_of_Dependents: float
    Education_Level: str
    Occupation: str
    Health_Score: float
    Location: str
    Policy_Type: str
    Previous_Claims: float
    Vehicle_Age: float
    Credit_Score: float
    Insurance_Duration: float
    Customer_Feedback: str
    Smoking_Status: str
    Exercise_Frequency: str
    Property_Type: str
    Policy_Start_Date: str

@app.get("/")
def home():
    return {"message": "✅ API جاهز للتنبؤ بالمبلغ التأميني"}

@app.post("/predict")
def predict(data: InsuranceData):
    try:
        # Convert input to DataFrame
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])

        # Rename columns to match training data
        rename_dict = {
            "Annual_Income": "Annual Income",
            "Marital_Status": "Marital Status",
            "Number_of_Dependents": "Number of Dependents",
            "Education_Level": "Education Level",
            "Health_Score": "Health Score",
            "Policy_Type": "Policy Type",
            "Previous_Claims": "Previous Claims",
            "Vehicle_Age": "Vehicle Age",
            "Credit_Score": "Credit Score",
            "Insurance_Duration": "Insurance Duration",
            "Customer_Feedback": "Customer Feedback",
            "Smoking_Status": "Smoking Status",
            "Exercise_Frequency": "Exercise Frequency",
            "Property_Type": "Property Type",
            "Policy_Start_Date": "Policy Start Date"
        }
        input_df = input_df.rename(columns=rename_dict)
        
        # Process date
        if 'Policy Start Date' in input_df.columns:
            try:
                input_df['Policy Start Date'] = pd.to_datetime(input_df['Policy Start Date'])
                input_df['year'] = input_df['Policy Start Date'].dt.year
                input_df['month'] = input_df['Policy Start Date'].dt.month
                input_df['day'] = input_df['Policy Start Date'].dt.day
            except Exception:
                # Default values for date error
                input_df['year'] = 2023
                input_df['month'] = 1
                input_df['day'] = 1
            finally:
                input_df.drop('Policy Start Date', axis=1, inplace=True)

        # Encode categorical variables
        for col in label_encoders:
            if col in input_df.columns:
                le = label_encoders[col]
                # Handle new values
                input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                
                # Add 'unknown' category if missing
                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')
                
                input_df[col] = le.transform(input_df[col])

        # Ensure expected columns
        expected_columns = [
            'Age', 'Gender', 'Annual Income', 'Marital Status', 'Number of Dependents',
            'Education Level', 'Occupation', 'Health Score', 'Location', 'Policy Type',
            'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
            'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type',
            'year', 'month', 'day'
        ]

        # Add missing columns with default value 0
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Make prediction
        prediction = model.predict(input_df)

        return {
            "prediction": float(prediction[0]),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)