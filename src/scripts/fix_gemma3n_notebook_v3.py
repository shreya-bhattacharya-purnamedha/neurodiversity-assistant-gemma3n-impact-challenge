# FIXED NOTEBOOK CELLS FOR GEMMA3N (3).ipynb
# Replace the problematic cells with these corrected versions

# ===== REPLACE CELL 15: Load Dataset =====
import json
from datasets import Dataset

# Load the dataset from the local JSON file
with open("/content/combined.json", 'r') as f:
    data = json.load(f)

print(f"📖 Loaded {len(data)} examples from combined.json")

# ===== REPLACE CELL 16: Convert to Intermediate Format =====
# Convert to intermediate format (Step 1)
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
            'score': 5.0
        }
        
        intermediate_data.append(intermediate_item)

print(f"✅ Converted {len(intermediate_data)} examples to intermediate format")

# Create HuggingFace Dataset
dataset = Dataset.from_list(intermediate_data)

print("📝 Example intermediate format:")
print(dataset[0])

# ===== REPLACE CELL 17: Apply Chat Template =====
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

dataset = dataset.map(formatting_prompts_func, batched=True)

print("📝 Formatted example (first 200 chars):")
print(dataset[0]["text"][:200] + "...")

# ===== REPLACE CELL 18: Verify Dataset =====
print(f"✅ Dataset ready for training with {len(dataset)} examples")
print(f"📊 Final dataset features: {dataset.features}")

# ===== DELETE CELLS 19-23 (they are redundant) =====
# Remove these cells as they duplicate the work already done above

# ===== CELL 24 and onwards remain the same =====
# The training cells (24+) can remain unchanged as they will work with the properly formatted dataset 