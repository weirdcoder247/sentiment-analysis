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
holdout_data_file_name = "reviews_test_4000.csv"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#------------------------------------------------------------------------------#


def data_import(input_path, holdout_data_file_name):

	holdout_df = pd.read_csv(input_path + holdout_data_file_name)

	return holdout_df


def data_transform(holdout_df):

	# Drop ID Column
	holdout_df.drop(["review_id"], axis=1, inplace=True)

	# Create binary sentiment variable
	holdout_df["sentiment_2"] = 0
	holdout_df.loc[holdout_df["sentiment"]=="positive", ["sentiment_2"]] = 1

	# Drop text sentiment column
	holdout_df.drop(["sentiment"], axis=1, inplace=True)

	# Rename Columns
	holdout_df.columns = ["review", "sentiment_label"]

	return holdout_df


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


def model_eval(classifier, holdout_df):

	# Predict Sentiment of reviews in holdout_df
	y = classifier(holdout_df.review.tolist())
	y = [1 if i['label'].lower() == 'positive' else 0 for i in y]

	# Actual Sentiment of reviews in holdout_df
	x = holdout_df.sentiment_label.tolist()

	# Model Prediction Accuracy on holdout_df
	print("holdout_df accuracy")
	print(confusion_matrix(y, x))
	print(accuracy_score(y, x))


def main():
	
	start_time = time()

	# Data Import
	holdout_df = data_import(input_path, holdout_data_file_name)

	# Data Transformation
	holdout_df = data_transform(holdout_df)

	# Create Sentiment Analysis Classifier
	classifier = model_import(model_name)
	
	# Predict and Evaluate Model
	model_eval(classifier, holdout_df)

	stop_time = time()
	print("Execution Time:", stop_time - start_time, "seconds")

if __name__ == "__main__":
	main()

