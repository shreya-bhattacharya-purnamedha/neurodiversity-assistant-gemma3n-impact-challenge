#!/usr/bin/env python3
"""
Simple Gemma3N 2B-it Fine-tuning Script
Inspired by Unsloth tutorial for neurodiversity assistant
"""

import json
import torch
from unsloth import FastModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def load_training_data(json_file_path):
    """Load training data from JSON file"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to chat format for Gemma3N
    formatted_data = []
    for item in data:
        # Assuming JSON has 'user_query' and 'assistant_response' fields
        # Adjust field names based on your actual JSON structure
        user_query = item.get('user_query', item.get('Sample_User_Query', ''))
        assistant_response = item.get('assistant_response', item.get('response', ''))
        
        if user_query and assistant_response:
            formatted_data.append({
                'messages': [
                    {'role': 'user', 'content': user_query},
                    {'role': 'assistant', 'content': assistant_response}
                ]
            })
    
    return formatted_data

def main():
    # Configuration
    JSON_FILE = "training_data.json"  # Your JSON file path
    MODEL_NAME = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
    OUTPUT_DIR = "./neurodiversity-assistant-finetuned"
    
    print("🚀 Loading Gemma3N 2B-it model...")
    
    # Load model with Unsloth optimizations
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
    )
    
    print("📖 Loading training data...")
    
    # Load and format training data
    training_data = load_training_data(JSON_FILE)
    dataset = Dataset.from_list(training_data)
    
    print(f"✅ Loaded {len(dataset)} training examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=1024,
        dataset_text_field="messages",
    )
    
    print("🎯 Starting fine-tuning...")
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    print("💾 Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Fine-tuning complete! Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 