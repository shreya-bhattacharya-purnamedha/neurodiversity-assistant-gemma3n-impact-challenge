# CORRECTED NOTEBOOK CELLS FOR GEMMA3N DATASET CONVERSION
# Replace the problematic cells in the notebook with these

# ===== CELL 15: Load Dataset =====
import json
from datasets import Dataset

# Load the combined.json dataset
json_file_path = "/content/combined.json"  # Adjust path as needed

with open(json_file_path, 'r') as f:
    data = json.load(f)

print(f"📖 Loaded {len(data)} examples from {json_file_path}")

# ===== CELL 16: Convert to Conversational Format =====
# Convert instruction/input/output format to Gemma 3N conversational format
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

# ===== CELL 17: Show Example =====
print("📝 Example conversation:")
print(dataset[0]["conversations"])

# ===== CELL 18: Apply Chat Template =====
# Apply Gemma 3N chat template formatting
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

dataset = dataset.map(formatting_prompts_func, batched=True)

# ===== CELL 19: Show Formatted Example =====
print("📝 Formatted example:")
print(dataset[0]["text"][:200] + "...")  # Show first 200 chars

# ===== CELL 20: Verify Dataset =====
print(f"✅ Dataset ready for training with {len(dataset)} examples")
print(f"📊 Dataset features: {dataset.features}") 