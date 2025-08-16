import requests
import time

# اختبار الصفحة الرئيسية
def test_home():
    print("\n=== Testing home endpoint ===")
    url = "http://localhost:8002/"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.json())
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# اختبار نقطة الصحة
def test_health():
    print("\n=== Testing health endpoint ===")
    url = "http://localhost:8002/health"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.json())
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# اختبار نقطة التنبؤ
def test_predict():
    print("\n=== Testing predict endpoint ===")
    url = "http://localhost:8002/predict"
    
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
    
    try:
        response = requests.post(url, json=sample_data)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.json())
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# اختبار شامل للتطبيق
def run_tests():
    max_retries = 5
    retry_delay = 2  # seconds
    
    print("Starting API tests...")
    
    for i in range(max_retries):
        try:
            # Test home endpoint
            if not test_home():
                print("Home test failed")
                continue
                
            # Test health endpoint
            if not test_health():
                print("Health test failed")
                continue
                
            # Test predict endpoint
            if not test_predict():
                print("Predict test failed")
                continue
                
            print("\n✅ All tests passed successfully!")
            return
            
        except Exception as e:
            print(f"\nAttempt {i+1}/{max_retries} failed: {str(e)}")
            if i < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    print("\n❌ Some tests failed after multiple attempts")

if __name__ == "__main__":
    run_tests()