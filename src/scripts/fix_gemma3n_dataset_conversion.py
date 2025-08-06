#!/usr/bin/env python3
"""
Fix for Gemma3N dataset conversion
Converts instruction/input/output format to proper conversational format
"""

import json
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

def load_and_convert_dataset(json_file_path):
    """Load the combined.json dataset and convert to Gemma 3N conversational format"""
    
    # Load the original dataset
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"📖 Loaded {len(data)} examples from {json_file_path}")
    
    # Convert to conversational format
    conversations = []
    for item in data:
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        assistant_output = item.get('output', '')
        
        if instruction and user_input and assistant_output:
            # Combine instruction and input for the user message
            user_message = f"{instruction}\n\nUser: {user_input}"
            
            # Create conversation in Gemma 3N format
            conversation = [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": user_message}]
                },
                {
                    "role": "model", 
                    "content": [{"type": "text", "text": assistant_output}]
                }
            ]
            
            conversations.append(conversation)
    
    print(f"✅ Converted {len(conversations)} conversations")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list([{"conversations": conv} for conv in conversations])
    
    return dataset

def apply_gemma3n_formatting(dataset, tokenizer):
    """Apply Gemma 3N chat template formatting"""
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        # Apply chat template and remove <bos> token for training
        texts = [
            tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ).removeprefix('<bos>') 
            for convo in convos
        ]
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset

def main():
    """Main function to demonstrate the fix"""
    
    # File paths
    json_file = "dataset/ift-fully-processed/combined.json"
    
    # Load and convert dataset
    print("🔄 Converting dataset to Gemma 3N format...")
    dataset = load_and_convert_dataset(json_file)
    
    # Show example before formatting
    print("\n📝 Example conversation before formatting:")
    print(dataset[0]["conversations"])
    
    # Note: In a real notebook, you would load the tokenizer here
    # tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    # dataset = apply_gemma3n_formatting(dataset, tokenizer)
    
    print(f"\n✅ Dataset ready for training with {len(dataset)} examples")
    print("💡 Use this dataset with the existing training pipeline in the notebook")

if __name__ == "__main__":
    main() 