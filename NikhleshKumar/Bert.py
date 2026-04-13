# -*- coding: utf-8 -*-
from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_csv("symptom.csv")

print(df.head())

print(df.shape)
print(df.columns)

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

import torch
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv("symptom.csv")

print(df.head())

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True,
    max_length=128
)

class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PlantDataset(train_encodings, train_labels)
test_dataset = PlantDataset(test_encodings, test_labels)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_)
)

!pip install -U transformers

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

from sklearn.metrics import accuracy_score

predictions = trainer.predict(test_dataset)

preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
print("Accuracy:", accuracy)

df = pd.read_csv("symptom.csv")

# SAVE ORIGINAL LABELS
original_labels = df["label"].copy()

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_id = logits.argmax().item()

    return le.inverse_transform([predicted_class_id])[0]

print(predict("yellow spots on leaves"))
print(predict("white powder on leaves"))

print("Training complete")
model.save_pretrained("bert_model")

model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

!zip -r bert_model.zip bert_model
from google.colab import files
files.download("bert_model.zip")

def chatbot():
    while True:
        text = input("Enter symptoms (or 'exit'): ")
        if text == "exit":
            break
        print("Predicted Disease:", predict(text))

chatbot()

chatbot()
