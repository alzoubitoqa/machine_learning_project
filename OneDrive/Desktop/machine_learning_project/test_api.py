import requests
import time

sample_data = {
    "Age": 35.0,
    "Gender": "Male",
    "Annual_Income": 65000.0,
    "Marital_Status": "Married",
    "Number_of_Dependents": 2.0,
    "Education_Level": "Bachelor's",
    "Occupation": "Employed",
    "Health_Score": 28.5,
    "Location": "Urban",
    "Policy_Type": "Premium",
    "Previous_Claims": 1.0,
    "Vehicle_Age": 5.0,
    "Credit_Score": 720.0,
    "Insurance_Duration": 3.0,
    "Customer_Feedback": "Good",
    "Smoking_Status": "No",
    "Exercise_Frequency": "Weekly",
    "Property_Type": "House",
    "Policy_Start_Date": "2023-05-15"
}

max_retries = 5
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        response = requests.post("http://localhost:8000/predict", json=sample_data, timeout=10)
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(response.json())
        break  # Exit loop if successful
    except requests.exceptions.ConnectionError:
        print(f"Connection failed (attempt {attempt+1}/{max_retries}). Is the server running?")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        break
else:
    print("❌ Failed to connect after multiple attempts. Please ensure:")
    print("- The FastAPI server is running")
    print("- You're using the command: uvicorn main:app --reload")
    print("- There are no errors in the server console")
    print("- Port 8000 is not blocked by firewall")