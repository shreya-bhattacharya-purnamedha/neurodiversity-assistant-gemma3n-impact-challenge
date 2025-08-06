#!/usr/bin/env python3
"""
CPU-only HuggingFace Transformers fine-tuning script (for pipeline testing)
Uses a small model (distilbert-base-uncased) and the same combined.json dataset.
No Unsloth, no 4-bit, just standard Trainer. Fast, for local CPU testing only.
"""

import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Use a small model for quick CPU testing
MODEL_NAME = "distilbert-base-uncased"
JSON_FILE = "dataset/ift-fully-processed/combined.json"
OUTPUT_DIR = "./test-finetuned-distilbert-cpu"

# Load and format data for sequence classification
# We'll treat the 'input' as text and 'output' as the label (for demo purposes)
def load_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    # For testing, just use the first 100 examples
    data = data[:100]
    samples = []
    for item in data:
        user_input = item.get('input', '')
        assistant_output = item.get('output', '')
        if user_input and assistant_output:
            samples.append({
                'text': user_input,
                'label': assistant_output[:128]  # Truncate label for demo
            })
    return samples

def main():
    if not os.path.exists(JSON_FILE):
        print(f"❌ Dataset not found at: {JSON_FILE}")
        return
    print("📖 Loading data...")
    samples = load_data(JSON_FILE)
    texts = [s['text'] for s in samples]
    labels = [s['label'] for s in samples]
    # For demo, treat each unique label as a class
    label2id = {l: i for i, l in enumerate(sorted(set(labels)))}
    id2label = {i: l for l, i in label2id.items()}
    y = [label2id[l] for l in labels]
    # Prepare HuggingFace Dataset
    dataset = Dataset.from_dict({'text': texts, 'label': y})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=64)
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # Split into train/test
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    # Training args (minimal for CPU)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        logging_steps=5,
        save_steps=10,
        save_total_limit=1,
        report_to=None,
        no_cuda=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    print("🚀 Starting CPU-only fine-tuning (this is just a pipeline test)...")
    trainer.train()
    print("✅ Done! Model saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()