#%% Setup & Data Preparation
import os
import torch
import json
import nltk
import math
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Ensure nltk data is available
nltk.download('punkt')

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load JSON dataset
json_file = "news_data_large.json"
with open(json_file, "r", encoding="utf-8") as f:
    news_data = json.load(f)

# Convert JSON into structured dataset
articles = [{"text": item["Story"], "summary": item["Headline"]} for item in news_data]  # Use Headline as summary
dataset = Dataset.from_list(articles)

# Load tokenizer and BART model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

#%% Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=150)
    inputs["labels"] = labels["input_ids"]
    return inputs

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset into train & validation (80% train, 20% validation)
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
validation_dataset = train_test_split["test"]

#%% Fine-Tuning BART Model
save_path = "./bart-news-finetuned"
os.makedirs(save_path, exist_ok=True)

# Training arguments (Optimized for Mac MPS & low-memory GPUs)
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none",
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Train the model
trainer.train()

# Save fine-tuned model
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Fine-tuning complete! Model saved to:", save_path)

# Load fine-tuned model for testing
fine_tuned_model = BartForConditionalGeneration.from_pretrained(save_path).to(device)
fine_tuned_model.eval()

#%% Evaluation: Summarization & Performance Metrics
def generate_summary(model, text):
    """Generate a summary using BART."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, num_return_sequences=1)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def compute_bleu(reference_text, generated_text):
    """Compute BLEU Score (higher = better quality)."""
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]
    generated_tokens = nltk.word_tokenize(generated_text.lower())
    return sentence_bleu(reference_tokens, generated_tokens)

def compute_rouge(reference_text, generated_text):
    """Compute ROUGE Score (higher = better summary retention)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

# Sample article for evaluation
sample_text = news_data[0]["Story"]
reference_summary = news_data[0]["Headline"]

# Generate summaries
pretrained_summary = generate_summary(model, sample_text)
fine_tuned_summary = generate_summary(fine_tuned_model, sample_text)

# Compute BLEU & ROUGE Scores
bleu_score = compute_bleu(reference_summary, fine_tuned_summary)
rouge1, rouge2, rougeL = compute_rouge(reference_summary, fine_tuned_summary)

#%% Print Results
print("\nPretrained Model Summary:\n", pretrained_summary)
print("\nFine-Tuned Model Summary:\n", fine_tuned_summary)
print(f"\nBLEU Score: {bleu_score:.4f}")
print(f"ROUGE-1: {rouge1:.4f} | ROUGE-2: {rouge2:.4f} | ROUGE-L: {rougeL:.4f}")

