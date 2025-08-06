#!/usr/bin/env python3
"""
Convert combined.json format to intermediate format for Gemma 3N
Step 1: instruction/input/output -> intermediate format with conversations
Step 2: Apply chat template to get final training format
"""

import json
from datasets import Dataset

def convert_ift_to_intermediate_format(json_file_path):
    """Step 1: Convert instruction/input/output to intermediate format"""
    
    # Load the original dataset
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"📖 Loaded {len(data)} examples from {json_file_path}")
    
    # Convert to intermediate format
    intermediate_data = []
    for item in data:
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        assistant_output = item.get('output', '')
        
        if instruction and user_input and assistant_output:
            # Combine instruction and input for the user message
            user_message = f"{instruction}\n\nUser: {user_input}"
            
            # Create intermediate format
            intermediate_item = {
                'conversations': [
                    {
                        'content': user_message,
                        'role': 'user'
                    },
                    {
                        'content': assistant_output,
                        'role': 'assistant'
                    }
                ],
                'source': 'neurodiversity-assistant',
                'score': 5.0  # High quality dataset
            }
            
            intermediate_data.append(intermediate_item)
    
    print(f"✅ Converted {len(intermediate_data)} examples to intermediate format")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(intermediate_data)
    
    return dataset

def apply_chat_template_formatting(dataset, tokenizer):
    """Step 2: Apply chat template to get final training format"""
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
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
    """Main function to demonstrate the complete conversion"""
    
    # File paths
    json_file = "dataset/ift-fully-processed/combined.json"
    
    # Step 1: Convert to intermediate format
    print("🔄 Step 1: Converting to intermediate format...")
    dataset = convert_ift_to_intermediate_format(json_file)
    
    # Show example of intermediate format
    print("\n📝 Example intermediate format:")
    print(dataset[0])
    
    # Step 2: Apply chat template (requires tokenizer)
    print("\n🔄 Step 2: Ready to apply chat template...")
    print("💡 In your notebook, run the formatting_prompts_func after loading the tokenizer")
    
    print(f"\n✅ Dataset ready with {len(dataset)} examples")
    print("📊 Dataset features:", dataset.features)

if __name__ == "__main__":
    main() 