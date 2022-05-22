# Import Packages
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import torch
from transformers import DefaultDataCollator
import tensorflow as tf
import datasets
from time import time

start_time = time()
# User Input
#------------------------------------------------------------------------------#
input_path = "~/Repositories/sentiment-analysis/01_data/sentiments_train_test_reviews/"
train_data_file_name = "reviews_training_26000.csv"
test_data_file_name = "reviews_test_4000.csv"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
output_path = "~/Repositories/sentiment-analysis/03_outputs/"
output_model_name = "distilbert_tuned_v1"
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


def tokenize_train_test(model_name, train_df, test_df):

	tokenizer = DistilBertTokenizer.from_pretrained(model_name, model_max_length=512)

	def tokenize_function(reviews):
		return tokenizer(reviews["review"], padding="max_length", truncation=True)

	train_df = datasets.Dataset.from_dict(train_df)
	test_df = datasets.Dataset.from_dict(test_df)

	train_test_datadict = datasets.DatasetDict({"train":train_df, "test":test_df})

	tokenized_datasets = train_test_datadict.map(tokenize_function, batched=True)

	#small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
	#small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
	train_dataset = tokenized_datasets["train"]
	eval_dataset = tokenized_datasets["test"]

	return train_dataset, eval_dataset


def tokenized_to_tf_dataset(train_dataset, eval_dataset):

	data_collator = DefaultDataCollator(return_tensors="tf")

	tf_train_dataset = train_dataset.to_tf_dataset(
		columns=["attention_mask", "input_ids", "review"],
		label_cols=["sentiment_label"],
		shuffle=True,
		collate_fn=data_collator,
		batch_size=8,
	)

	tf_validation_dataset = eval_dataset.to_tf_dataset(
		columns=["attention_mask", "input_ids", "review"],
		label_cols=["sentiment_label"],
		shuffle=False,
		collate_fn=data_collator,
		batch_size=8,
	)

	return tf_train_dataset, tf_validation_dataset


def model_train_and_save(model_name, tf_train_dataset, tf_validation_dataset, output_path, output_model_name):

	model = TFDistilBertForSequenceClassification.from_pretrained(model_name)

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
		    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		    metrics=tf.metrics.SparseCategoricalAccuracy())

	with tf.device("/gpu:0"):
		model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=1)

	model.save(output_path + output_model_name)


def main():
	
	start_time = time()

	# Data Import
	train_df, test_df = data_import(input_path, train_data_file_name, test_data_file_name)

	# Data Transformation
	train_df, test_df = data_transform(train_df, test_df)

	# Tokenize Datasets
	train_dataset, eval_dataset = tokenize_train_test(model_name, train_df, test_df)

	# Convert tokenized data to tensors
	tf_train_dataset, tf_validation_dataset = tokenized_to_tf_dataset(train_dataset, eval_dataset)

	# Train and save the model
	model_train_and_save(model_name, tf_train_dataset, tf_validation_dataset, output_path, output_model_name)

	stop_time = time()
	print("Execution Time:", stop_time - start_time, "seconds")

if __name__ == "__main__":
	main()

