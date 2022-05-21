# Install Transformers library;
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import os
import torch
from sklearn.metrics import classification_report, auc, accuracy_score, confusion_matrix, f1_score
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

# print(train_df.head())
# print(test_df.head())

# sentiment_pipeline = pipeline("sentiment-analysis")

# tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}

# print(sentiment_pipeline(train_df.review.head(20).tolist()))


# Approach 2 using AutoModel & AutoTokenizer
# model_name = "bert-base-cased"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
device = 0 if torch.cuda.is_available() else -1
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device, padding=True, truncation=True)

# Model Prediction
y = classifier(test_df.review.tolist())
y = [1 if i['label'].lower() == 'positive' else 0 for i in y]

# Model Evaluation
x = test_df.sentiment_label.tolist()

print(confusion_matrix(y, x))
print(accuracy_score(y, x))

stop_time = time()

print("Execution Time:", stop_time - start_time, "seconds")
