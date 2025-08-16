from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, JSON, Float, DateTime, Integer, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
from dotenv import load_dotenv
import sys
from pydantic import BaseModel
import json

# وظيفة مساعدة للترميز الآمن
def safe_label_transform(encoder, values):
    transformed = []
    for val in values:
        if val in encoder.classes_:
            transformed.append(encoder.transform([val])[0])
        else:
            if 'unknown' in encoder.classes_:
                transformed.append(encoder.transform(['unknown'])[0])
            else:
                most_common = encoder.transform([encoder.classes_[0]])[0]
                transformed.append(most_common)
    return transformed

# Load environment variables
load_dotenv()

# Initialize FastAPI app with documentation
app = FastAPI(
    title="خدمة تنبؤات التأمين",
    description="API للتنبؤ بمبالغ التأمين باستخدام نموذج تعلم الآلة",
    version="1.0.0",
    docs_url="/docs",  # تفعيل واجهة المستندات
    redoc_url=None,    # تعطيل الواجهة البديلة
)

# 1. Load the model
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# 2. Load label encoders
LABEL_ENCODERS_PATH = os.getenv("LABEL_ENCODERS_PATH", "label_encoders.joblib")
try:
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    print("✅ Label encoders loaded successfully")
    print("Encoded columns:", list(label_encoders.keys()))
    # طباعة تفاصيل الترميز
    for col, encoder in label_encoders.items():
        print(f"  - {col}: {list(encoder.classes_)}")
except Exception as e:
    print(f"❌ Error loading label encoders: {e}")
    sys.exit(1)

# 3. Database setup
DATABASE_URL = "sqlite:///./predictions.db"
try:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    print("✅ Database connection established")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    sys.exit(1)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions_of_ml_service_project"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(JSON)
    prediction = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        return False

if not create_tables():
    sys.exit(1)

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

# نقطة النهاية للصفحة الرئيسية
@app.get("/")
async def home():
    return {
        "message": "مرحبًا بك في خدمة تنبؤات التأمين",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        },
        "documentation": "http://localhost:8002/docs"
    }

@app.post("/predict")
async def predict(data: InsuranceData):
    try:
        # استخدام model_dump() بدلاً من dict()
        input_dict = data.model_dump()
        
        input_df = pd.DataFrame([input_dict])
        
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
        
        # طباعة البيانات قبل المعالجة
        print("\n=== البيانات قبل المعالجة ===")
        print(input_df)
        
        if 'Policy Start Date' in input_df.columns:
            try:
                input_df['Policy Start Date'] = pd.to_datetime(input_df['Policy Start Date'])
                input_df['year'] = input_df['Policy Start Date'].dt.year
                input_df['month'] = input_df['Policy Start Date'].dt.month
                input_df['day'] = input_df['Policy Start Date'].dt.day
            except Exception:
                input_df['year'] = 2023
                input_df['month'] = 1
                input_df['day'] = 1
            finally:
                input_df.drop('Policy Start Date', axis=1, inplace=True)

        # الترميز باستخدام الوظيفة الآمنة
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                
                print(f"\nترميز العمود: {col}")
                print(f"القيم الأصلية: {input_df[col].values}")
                print(f"فئات المرمِّز: {le.classes_}")
                
                input_df[col] = safe_label_transform(le, input_df[col])
                print(f"القيم بعد الترميز: {input_df[col].values}")

        expected_columns = [
            'Age', 'Gender', 'Annual Income', 'Marital Status', 'Number of Dependents',
            'Education Level', 'Occupation', 'Health Score', 'Location', 'Policy Type',
            'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
            'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type',
            'year', 'month', 'day'
        ]

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        
        # التحقق من أنواع البيانات
        for col in input_df.columns:
            if not pd.api.types.is_numeric_dtype(input_df[col]):
                print(f"تحذير: العمود {col} ليس رقميًا، القيم: {input_df[col].values}")
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # طباعة البيانات بعد المعالجة
        print("\n=== البيانات بعد المعالجة ===")
        print(input_df)
        print("أنواع البيانات:")
        print(input_df.dtypes)

        prediction = float(model.predict(input_df)[0])
        
        db = SessionLocal()
        try:
            db_record = PredictionRecord(
                input_data=input_dict,
                prediction=prediction
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
            return {
                "prediction": prediction,
                "db_id": db_record.id,
                "status": "success"
            }
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            db.close()

    except Exception as e:
        import traceback
        print(f"حدث خطأ: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        # فحص قاعدة البيانات
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # فحص النموذج
        if not hasattr(model, "predict"):
            raise Exception("النموذج لا يحتوي على دالة التنبؤ")
        
        # فحص المرمّزات
        if not label_encoders:
            raise Exception("لم يتم تحميل المرمّزات")
        
        return {
            "status": "سليم",
            "database": "متصل",
            "model": "محمل",
            "label_encoders": "محملة",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "غير سليم",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.getenv("PORT", 8002))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)