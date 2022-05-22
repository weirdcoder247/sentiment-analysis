# Install Transformers library
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time


start_time = time()
# User Input
input_path = "~/Repositories/sentiment-analysis/01_data/sentiments_train_test_reviews/"
train_data_file_name = "reviews_training_26000.csv"
test_data_file_name = "reviews_test_4000.csv"

# Import Data
train_df = pd.read_csv(input_path + train_data_file_name)
test_df = pd.read_csv(input_path + test_data_file_name)


# Data Tranformation
train_df.drop(["review_id"], axis=1, inplace=True)
test_df.drop(["review_id"], axis=1, inplace=True)

train_df["sentiment_2"] = 0
test_df["sentiment_2"] = 0

train_df.loc[train_df["sentiment"]=="positive", ["sentiment_2"]] = 1
test_df.loc[test_df["sentiment"]=="positive", ["sentiment_2"]] = 1

train_df.drop(["sentiment"], axis=1, inplace=True)
test_df.drop(["sentiment"], axis=1, inplace=True)

train_df.columns = ["review", "sentiment_label"]
test_df.columns = ["review", "sentiment_label"]

# AutoModel & AutoTokenizer
# model_name = "bert-base-cased"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
device = 0 if torch.cuda.is_available() else -1
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device, padding=True, truncation=True)

# Model Prediction & Evaluation
# test_df
y = classifier(test_df.review.tolist())
y = [1 if i['label'].lower() == 'positive' else 0 for i in y]

x = test_df.sentiment_label.tolist()

print("test_df accuracy")
print(confusion_matrix(y, x))
print(accuracy_score(y, x))

# train_df
y = classifier(train_df.review.tolist())
y = [1 if i['label'].lower() == 'positive' else 0 for i in y]

x = train_df.sentiment_label.tolist()

print("train_df_accuracy")
print(confusion_matrix(y, x))
print(accuracy_score(y, x))

stop_time = time()

print("Execution Time:", stop_time - start_time, "seconds")
