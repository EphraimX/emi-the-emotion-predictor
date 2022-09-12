from fastapi import FastAPI, Request, File, UploadFile, Form    
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
    username: str
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

    # username = request.query_params['username']
    # feelings = request.query_params['requests']

    # print(username, feelings)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/emotions')
async def emotionsClassifer(request: Request):

    username: str = request.query_params['username']
    word_request: str = request.query_params['request']
    
    emotion = predict(word=word_request)
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

    if response == "Joy":
        emotion_response = "Joyous"
        return templates.TemplateResponse("joy.html",
                                     {"request":request,
                                      "username" : username,
                                       "emotion" : emotion_response
                                    })
    

    if response == "Sadness":
        emotion_response = "Sad"
        return templates.TemplateResponse("sad.html",
                                     {"request":request,
                                      "username" : username,
                                       "emotion" : emotion_response
                                    })
    

    if response == "Anger":
        emotion_response = "Angry"
        return templates.TemplateResponse("anger.html",
                                     {"request":request,
                                      "username" : username,
                                       "emotion" : emotion_response
                                    })
    

    if response == "Fear":
        emotion_response = "Afraid"
        return templates.TemplateResponse("fear.html",
                                     {"request":request,
                                      "username" : username,
                                       "emotion" : response
                                    })
    

    if response == "Surprise":
        emotion_response = "Suprised"
        return templates.TemplateResponse("surprise.html",
                                     {"request":request,
                                      "username" : username,
                                       "emotion" : emotion_response
                                    })
    

    if response == "Love":
        emotion_response = "Loved"
        return templates.TemplateResponse("love.html",
                                     {"request":request,
                                      "username" : username,
                                       "emotion" : emotion_response
                                    })
    



# "endpoint": "https://srvre2.deta.dev",