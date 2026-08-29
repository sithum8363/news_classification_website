from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import time
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
import json
import os

all_data = []

category = "General"

for page in range(1, 20):
    url = f"https://hirunews.lk/api/fetch_en_news.php?page={page}&category={category}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if not data:
            print("No data on page:", page)
            break

        for item in data:
            item["category"] = category
            all_data.append(item)

        print("Collected page:", page, "items:", len(data))

    time.sleep(1)
df = pd.DataFrame(all_data)


save_path = "bsithum8363/new_classifiaction_bert"

bert_model = BertForSequenceClassification.from_pretrained(
    save_path,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    save_path,
    local_files_only=True
)

bert_model.eval()

def predict_shap(texts):
    results = []

    for text in texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        results.append(probs.cpu().numpy()[0])

    return np.array(results)

# Function for prediction
def predict_news(text):
    probs = predict_shap([text])[0]

    predicted_class = np.argmax(probs)

    print("Class Probabilities:")
    print(probs)

    print("\nPredicted Class:", predicted_class)

    return predicted_class
df_value={ "eng_title": df["eng_title"],
"eng_story":df["eng_story"]
}
df = pd.DataFrame(df_value)
json_data = df.to_json(orient="records")
obj = json.loads(json_data)
# Select news article
count=0
news_list=[]
while count<100:
  
  value = str(obj[count]["eng_title"]) + " " + str(obj[count]["eng_story"])
  prediction = predict_news(value)
  print(value)
  df_va={  "category":int(prediction),
    "headline": df["eng_title"][count],
    "summary":df["eng_story"][count],
    "source": "Hiru News",
}
  news_list.append(df_va)


  count+=1
print(df_va)
app = Flask(__name__)
CORS(app)




@app.route('/')
def home():
    return "News API Running"
@app.route('/news/<int:id>')
def get_article(id):

    return jsonify({
        "id": id,
        "headline": "Government announces new project",
        "content": "This is the full news article."
    })
@app.route('/news', methods=['POST'])
def news():

    return jsonify(news_list)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000"))
