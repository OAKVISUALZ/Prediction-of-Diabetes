import json
import boto3
import joblib
import os
import numpy as np

# Initialize S3 client
s3 = boto3.client('s3')

# Environment variables (set these in AWS Lambda console)
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'my-diabetes-model-bucket')
MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME', 'diabetes_predictor.pkl')

def load_model_from_s3():
    """Downloads the model from S3 to the /tmp directory (the only writable path in Lambda)"""
    download_path = f'/tmp/{MODEL_FILE_NAME}'
    
    # Only download if we haven't already (caching for warm starts)
    if not os.path.exists(download_path):
        print(f"Downloading {MODEL_FILE_NAME} from S3...")
        s3.download_file(BUCKET_NAME, MODEL_FILE_NAME, download_path)
    
    return joblib.load(download_path)

# Global variable to hold the model (caches it for subsequent requests)
model = None

def lambda_handler(event, context):
    global model
    
    # 1. Load the model if it's not ready
    if model is None:
        model = load_model_from_s3()
    
    try:
        # 2. Parse the incoming JSON body from API Gateway
        body = json.loads(event['body'])
        
        # Extract features (ensure they match your training columns order)
        # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        features = np.array([[
            body['Pregnancies'],
            body['Glucose'],
            body['BloodPressure'],
            body['SkinThickness'],
            body['Insulin'],
            body['BMI'],
            body['DiabetesPedigreeFunction'],
            body['Age']
        ]])
        
        # 3. Make Prediction
        prediction = model.predict(features)
        result = int(prediction[0]) # 0 or 1
        
        # 4. Return JSON response
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({
                'prediction': result,
                'message': 'Diabetic' if result == 1 else 'Non-Diabetic'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
