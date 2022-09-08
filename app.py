from fastapi import FastAPI
import mlflow
import re
from sentence_transformers import SentenceTransformer


app = FastAPI()

mlflow.set_tracking_uri('sqlite:///mlflow/runs_info.db')

model_name = "Histogram Gradient Boosting Classifier - 2022-09-08"
model_version = 1

model_path = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_path)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def predict(word):
    encoder_request = encoder.encode(word)
    model_prediction = model.predict(encoder_request)

    return model_prediction


@app.get("/")
async def home():

    response = {
        "message" : "Hello"
    }

    return response


@app.post('/emotions')
async def emotionsClassifer(request):
    response = predict(word=request)
    return response

    