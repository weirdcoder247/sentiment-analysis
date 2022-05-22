# Import Packages
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import torch
from transformers import DefaultDataCollator
import tensorflow as tf
import datasets

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

# Convert train_df and test_df to tensors
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name, model_max_length=512)

def tokenize_function(reviews):
    return tokenizer(reviews["review"], padding="max_length", truncation=True)

train_df = datasets.Dataset.from_dict(train_df)
test_df = datasets.Dataset.from_dict(test_df)

train_test_datadict = datasets.DatasetDict({"train":train_df, "test":test_df})

tokenized_datasets = train_test_datadict.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

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


# device = 0 if torch.cuda.is_available() else -1
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy())

with tf.device("/gpu:0"):
    model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=1)

model.save("/home/charon/Repositories/sentiment-analysis/03_outputs/distilbert_tuned_v1")

# Load Tuned Model
model = tf.keras.models.load_model('/home/charon/Repositories/sentiment-analysis/03_outputs/distilbert_tuned_v1')
