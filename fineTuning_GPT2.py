#%% Setup & Data Preparation
import os
import torch
import math
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

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
articles = [{"text": item["Story"]} for item in news_data]  # Use full news article
dataset = Dataset.from_list(articles)

# Load tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Set padding token (GPT-2 does not have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Ensure labels are set
    return tokenized

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset into train & validation (80% train, 20% validation)
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
validation_dataset = train_test_split["test"]

#%% Fine-Tuning GPT-2 Model
save_path = "./gpt2-news-finetuned"
os.makedirs(save_path, exist_ok=True)

# Training arguments (Optimized for Mac MPS & low-memory GPUs)
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=2,  # Reduce batch size to prevent memory crashes
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Helps maintain an effective batch size
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none",  # Disable Weights & Biases (wandb) logging
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
fine_tuned_model = GPT2LMHeadModel.from_pretrained(save_path).to(device)
fine_tuned_model.eval()

#%% Evaluation: Text Generation & Performance Metrics
# Load fine-tuned GPT-2 model & tokenizer (Skip training)
save_path = "./gpt2-news-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(save_path)
fine_tuned_model = GPT2LMHeadModel.from_pretrained(save_path).to(device)
fine_tuned_model.eval()

#%% Evaluation: Text Generation & Performance Metrics
def generate_text(model, prompt="TSMC announced a new chip technology"):
    """Generate text using GPT-2 with proper attention mask."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Fix MPS issue: Ensure token IDs remain int64
    inputs["input_ids"] = inputs["input_ids"].to(torch.int64)

    # If using MPS, force CPU execution for generation (to avoid isin() bug)
    execution_device = "cpu" if device.type == "mps" else device
    inputs = {k: v.to(execution_device) for k, v in inputs.items()}

    # Generate text with explicit attention mask and pad token
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_perplexity(model, tokenizer, text):
    """Compute Perplexity Score (lower = better model performance)."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Fix MPS issue: Ensure token IDs are int64 and use correct device
    inputs["input_ids"] = inputs["input_ids"].to(torch.int64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())

def compute_bleu_score(reference_text, generated_text):
    """Compute BLEU Score (higher = better generated text quality)."""
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]
    generated_tokens = nltk.word_tokenize(generated_text.lower())
    return sentence_bleu(reference_tokens, generated_tokens)

# Comparison: Pretrained vs. Fine-Tuned GPT-2
sample_text = "TSMC announced a new chip technology"

# Generate text with fine-tuned model
fine_tuned_output = generate_text(fine_tuned_model, sample_text)

# Compute BLEU Score
reference_text = "TSMC announced their latest semiconductor innovation, focusing on advanced node technology."
bleu_score = compute_bleu_score(reference_text, fine_tuned_output)

# Compute Perplexity
ppl = compute_perplexity(fine_tuned_model, tokenizer, sample_text)

# Print Results
print("\nFine-Tuned Model Output:\n", fine_tuned_output)
print(f"\nBLEU Score: {bleu_score:.4f}")
print(f"Perplexity: {ppl:.2f}")
# %%
