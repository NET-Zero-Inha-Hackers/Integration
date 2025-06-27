from typing import Union

from fastapi import FastAPI

import tensorflow as tf
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
from transformers import TFBertModel
from fastapi import Body
import os

api_key = os.getenv("API_KEY", "")
print(f"API_KEY: {api_key}")


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')


class KoBertModel(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(KoBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='sigmoid',)

    def call(self, inputs):
        input_ids, attention_mask = inputs[:, 0], inputs[:, 1]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        x = self.dropout1(pooled_output)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

model = KoBertModel(num_classes=2)
print(model(np.zeros((1, 2, 128), dtype=np.int32), training=False))  # Dummy forward pass to build the model
model.load_weights('kobert_model_weights.h5')



app = FastAPI()

# Define the request body(json format {"prompt": "your text here"})
@app.post("/predict")
async def predict(payload: dict = Body(...)):
    if "api_key" not in payload or payload["api_key"] != api_key:
        return {"error": "Invalid API key"}
    text = payload.get("prompt", "")
    if not text:
        return {"error": "No prompt provided"}

    X_tokenized = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='np')
    X_id = X_tokenized['input_ids'] # (10000, 128)
    X_mask = X_tokenized['attention_mask'] # (10000, 128)

    X = np.stack([X_id, X_mask], axis=1) # (10000, 2, 128)
    predictions = model(X, training=False)
    predictions = tf.sigmoid(predictions).numpy()

    return {"predictions": {"gpt-4o": predictions[0][0].item(), "gpt-4o-mini": predictions[0][1].item()}}
