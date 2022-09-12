from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import mlflow


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


mlflow.set_tracking_uri('sqlite:///mlflow/runs_info.db')

model_name = "Histogram Gradient Boosting Classifier - 2022-09-08"
model_version = 1

model_path = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_path)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

templates = Jinja2Templates(directory="views")


class WordRequest(BaseModel):
    request : str


def predict(word):
    encoded_request = encoder.encode(word)
    model_prediction = model.predict([encoded_request])

    return model_prediction


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    response = {
        "message" : " Hi, my name is Emi, and I can tell what you feel by telling me how you feel. \n Wanna give it a try? "
    }

    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/emotions')
async def emotionsClassifer(request: Request, word_request: WordRequest):
    
    emotion = predict(word=word_request.request)
    emotion = emotion[0]
    
    emotion_class = {
        0: "Joy",
        1: "Sadness",
        2: "Anger",
        3: "Love",
        4: "Fear",
        5: "Surprise"
    }

    response = {
        "emotions" : emotion_class[emotion]
    }

    response = response["emotions"]

    # return response
    return templates.TemplateResponse("emotions.html", {"request":request, "emotion" : response})


# "endpoint": "https://srvre2.deta.dev",