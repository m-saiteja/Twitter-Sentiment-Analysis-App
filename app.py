import json
import os.path
import numpy as np
from flask import Flask, render_template, request
from db import create_db, query_all, insert_tweet

import torch
from data_prep import tokenize_tweet
from bert_classifier import BertClassifierModel

model = torch.load("bert_model.pt", map_location='cpu')
model.to("cpu")
model.eval()
print("Model Loaded and set to eval ...")


if not os.path.isfile("twitter_sentiment_analysis.db"):
    res = create_db()
    print(res)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    """
    Home route
    """
    if request.method == "POST":
        # include code for feedback
        fb = request.form.to_dict()
        if "correct" in fb.keys():
            feedback = "CORRECT"
        elif "wrong" in fb.keys():
            feedback = "WRONG"
        data = fb["old_data"].replace("'", '"')
        data = json.loads(data)
        pred = data["pred"]
        tweet = data["tweets"]
        insert_tweet(tweet, pred, feedback)
    return render_template("home.html")


@app.route("/data")
def data():
    """
    Data route
    """
    data_obj = query_all()
    data = [i for i in data_obj]
    return render_template("data.html", data=data)
    


@app.route("/result", methods=["POST"])
def result():
    """
    result route
    """
    user_tweet = request.form["tweet"]
    input_ids, attn_mask = tokenize_tweet(user_tweet)
    pred_probs = model(input_ids, attn_mask).detach().numpy()
    prediction = np.argmax(pred_probs)
    if prediction == 0:
        pred = "NEGATIVE"
    elif prediction == 1:
        pred = "NEUTRAL"
    else:
        pred = "POSITIVE"

    data = dict()
    data["pred"] = pred
    data["tweets"] = user_tweet
    return render_template("result.html", data=data)


if __name__ == "__main__":
    app.run(debug=False)
