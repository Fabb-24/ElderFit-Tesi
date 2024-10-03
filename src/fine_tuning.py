from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the tokenizer and model
model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the dataset
from datasets import load_dataset
dataset = load_dataset("imdb")