from flask import Flask, request, jsonify
import torch
import pickle
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = BertForSequenceClassification.from_pretrained("./bert_model")
tokenizer = BertTokenizer.from_pretrained("./bert_model")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model.eval()

# =========================
# PREDICTION FUNCTION
# =========================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    label = label_encoder.inverse_transform([pred.item()])[0]

    return label, confidence.item()

# =========================
# API ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "No input provided"}), 400

    label, confidence = predict(text)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 3)
    })

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)