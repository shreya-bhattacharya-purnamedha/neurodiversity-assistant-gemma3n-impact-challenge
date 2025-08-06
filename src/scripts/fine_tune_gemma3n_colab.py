#!/usr/bin/env python3
"""
Gemma3N 2B-it Fine-tuning Script for Google Colab
Optimized for GPU acceleration
Neurodiversity Assistant Project
"""

import json
import torch
import os
from unsloth import FastModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def load_training_data(json_file_path):
    """Load training data from combined.json and format for Gemma3N"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to chat format for Gemma3N
    formatted_data = []
    for item in data:
        # Extract instruction, input, and output from the combined.json format
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        assistant_output = item.get('output', '')
        
        if instruction and user_input and assistant_output:
            # Combine instruction and input for the user message
            user_message = f"{instruction}\n\nUser: {user_input}"
            
            formatted_data.append({
                'messages': [
                    {'role': 'user', 'content': user_message},
                    {'role': 'assistant', 'content': assistant_output}
                ]
            })
    
    return formatted_data

def main():
    # Configuration for Google Colab
    JSON_FILE = "dataset/ift-fully-processed/combined.json"
    MODEL_NAME = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
    OUTPUT_DIR = "./neurodiversity-assistant-finetuned-colab"
    
    # Check if dataset exists
    if not os.path.exists(JSON_FILE):
        print(f"❌ Dataset not found at: {JSON_FILE}")
        print("Please ensure the combined.json file exists in the dataset/ift-fully-processed/ directory")
        return
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("🚀 Loading Gemma3N 2B-it model for Colab training...")
    
    # Load model with Unsloth optimizations - optimized for GPU
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
    
    # Training arguments optimized for Google Colab GPU
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # More epochs for GPU training
        per_device_train_batch_size=4,  # Larger batch size for GPU
        gradient_accumulation_steps=4,  # Reduced since we have larger batch size
        learning_rate=2e-4,  # Standard learning rate for GPU
        fp16=True,  # Enable fp16 for GPU training
        bf16=False,  # Keep bf16 disabled for compatibility
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        save_total_limit=3,
        gradient_checkpointing=False,  # Disable for GPU efficiency
        dataloader_pin_memory=True,  # Enable for GPU training
        remove_unused_columns=False,
        report_to=None,  # Disable wandb for Colab
        # Colab-specific optimizations
        dataloader_num_workers=2,
        group_by_length=True,  # Group similar length sequences for efficiency
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
    
    print("🎯 Starting fine-tuning on Google Colab...")
    print("⚡ GPU acceleration enabled for faster training")
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    print("💾 Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Fine-tuning complete! Model saved to: {OUTPUT_DIR}")
    print("🎉 Your neurodiversity assistant model is ready!")
    
    # Optional: Save to Google Drive for persistence
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Copy model to Google Drive
        import shutil
        drive_path = "/content/drive/MyDrive/neurodiversity-assistant-model"
        shutil.copytree(OUTPUT_DIR, drive_path, dirs_exist_ok=True)
        print(f"💾 Model also saved to Google Drive: {drive_path}")
    except ImportError:
        print("📝 Note: Google Drive integration not available (not running in Colab)")

if __name__ == "__main__":
    main() 