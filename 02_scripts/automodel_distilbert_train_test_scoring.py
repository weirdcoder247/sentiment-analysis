# Import Packages
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time


# User Input
#------------------------------------------------------------------------------#
input_path = "~/Repositories/sentiment-analysis/01_data/sentiments_train_test_reviews/"
train_data_file_name = "reviews_training_26000.csv"
test_data_file_name = "reviews_test_4000.csv"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#------------------------------------------------------------------------------#


def data_import(input_path, train_data_file_name, test_data_file_name):

	train_df = pd.read_csv(input_path + train_data_file_name)
	test_df = pd.read_csv(input_path + test_data_file_name)

	return train_df, test_df


def data_transform(train_df, test_df):

	# Drop ID Column
	train_df.drop(["review_id"], axis=1, inplace=True)
	test_df.drop(["review_id"], axis=1, inplace=True)

	# Create binary sentiment variable
	train_df["sentiment_2"] = 0
	test_df["sentiment_2"] = 0
	train_df.loc[train_df["sentiment"]=="positive", ["sentiment_2"]] = 1
	test_df.loc[test_df["sentiment"]=="positive", ["sentiment_2"]] = 1

	# Drop text sentiment column
	train_df.drop(["sentiment"], axis=1, inplace=True)
	test_df.drop(["sentiment"], axis=1, inplace=True)

	# Rename Columns
	train_df.columns = ["review", "sentiment_label"]
	test_df.columns = ["review", "sentiment_label"]

	return train_df, test_df


def model_import(model_name):

	# 0 for GPU and -1 for CPU
	device = 0 if torch.cuda.is_available() else -1

	# Import pre-trained model and tokenizer
	model = AutoModelForSequenceClassification.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

	# Create a sentiment analysis classifier pipeline
	classifier = pipeline(
		"sentiment-analysis",
		model=model,
		tokenizer=tokenizer,
		device=device,
		padding=True,
		truncation=True
	)

	return classifier


def model_eval(classifier, train_df, test_df):

	# Predict Sentiment of reviews in test_df
	y = classifier(test_df.review.tolist())
	y = [1 if i['label'].lower() == 'positive' else 0 for i in y]

	# Actual Sentiment of reviews in test_df
	x = test_df.sentiment_label.tolist()

	# Model Prediction Accuracy on test_df
	print("test_df accuracy")
	print(confusion_matrix(y, x))
	print(accuracy_score(y, x))

	# Predict Sentiment of reviews in train_df
	y = classifier(train_df.review.tolist())
	y = [1 if i['label'].lower() == 'positive' else 0 for i in y]

	# Actual Sentiment of reviews in train_df
	x = train_df.sentiment_label.tolist()

	# Model Prediction Accuracy on train_df
	print("train_df_accuracy")
	print(confusion_matrix(y, x))
	print(accuracy_score(y, x))


def main():
	
	start_time = time()

	# Data Import
	train_df, test_df = data_import(input_path, train_data_file_name, test_data_file_name)

	# Data Transformation
	train_df, test_df = data_transform(train_df, test_df)

	# Create Sentiment Analysis Classifier
	classifier = model_import(model_name)
	
	# Predict and Evaluate Model
	model_eval(classifier, train_df, test_df)

	stop_time = time()
	print("Execution Time:", stop_time - start_time, "seconds")

if __name__ == "__main__":
	main()

