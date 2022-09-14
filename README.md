# Emi The Emotion Predictor

Hello and thank you for visiting this GitHub repository. This project was created as MLops ZoomCamp final capstone project, where I learned the art of building, deploying, testing, and monitoring machine learning models in production.

## Problem Statement.

The goal of this project was to assist people in identifying how they felt at any given time. Sometimes we feel a certain way but are unsure of what that feeling is. This project clarifies your feelings by giving them a name.

## The Process.

- Data, as with all machine learning models, is the foundation for everything. This project's data was obtained from [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp). It describes how a person feels and categorizes it into six categories: Joy, Anger, Fear, Sadness, Surprise, and Love. Although the data was reasonably clean, I still needed to correct the text, remove stopwords, and apply lemmatization. I also converted the words to vectors using HuggingFace's sentence-transformers library because machine learning algorithms mostly work with numeric values. I also assigned numerical values to the emotion classes (Joy, Love, and so on).

- Next, I trained the data with Scikit-Learn's Histogram Gradient Boosting Classifier, achieving a baseline accuracy average score of 61%. I kept getting the error 'ValueError: Setting an array element with a sequence' while trying to train the model. The algorithm was unable to match the vectors array to its specific class in this case. After many trials, I discovered that the solution was to make each array set an item in a list: '[x for x in train df]', which solved the problem.

- Model tracking, registry, and serving came next. I used [MLflow](https://www.mlflow.org) to track my model's hyperparameters and metrics over time to measure the performance of my model as I built it. This step in the process ensures that machine learning models can be easily replicated. For more information on the importance of model versioning in machine learning, see this [article](https://www.unbox.ai/blog/post/the-importance-of-model-versioning-in-machine-learning). Following that, the model was saved and served from the MLflow model registry. A model registry allows you to easily version your models and identify their current state (staging, production, and archived).

- Next, I used [Prefect](https://www.prefect.io) to orchestrate my model. The goal of orchestration is to create a workflow for your model, starting with data collection and ending with the model serving in production. This workflow ensures that each step of the building process runs smoothly, and if any part of the workflow fails, Prefect logs the error and attempts to restart the process.

- Finally, I ran the model on [Railway] (railway.app). This procedure was relatively simple. I set up my requirements.txt file, and Railway took care of the rest. At this point, I encountered the error 'Error: Deployment is not listening on the correct port.' The solution was to go to Railway's Variables tab and create an environment variable named PORT with the value 5000. (you can use any value here). Also, in the Railway settings tab, modify the start command to 'uvicorn main:app —host 0.0.0.0 —port 5000'. This solved the problem.

## Check it out.

The final deployment can be found [here](https://emi-the-emotion-predictor-production.up.railway.app). Thank you for sticking with me until the end. If you find this repo interesting, please give it a star.

![Emi's Home Page](https://imgur.com/7d19Kv1.jpeg)

## Acknowledgement

A special thank you to [DataTalks.Club](https://datatalks.club) and everyone from Prefect and Evidently.AI who collaborated to make this course a success. Cheers to more learnings.
