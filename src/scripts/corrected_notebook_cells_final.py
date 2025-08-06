# FINAL CORRECTED NOTEBOOK CELLS - FIXES CHAT TEMPLATE ERROR
# Replace the problematic cells in your Gemma3N notebook

# ===== CELL 15: Load Dataset =====
import json
from datasets import Dataset

# Load the dataset from the local JSON file
with open("/content/combined.json", 'r') as f:
    data = json.load(f)

print(f"📖 Loaded {len(data)} examples from combined.json")

# ===== CELL 16: Set up Chat Template FIRST (CRITICAL FIX) =====
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

print("✅ Chat template applied to tokenizer")

# ===== CELL 17: Convert to Intermediate Format =====
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

# ===== CELL 18: Apply Chat Template (NOW it will work) =====
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

# ===== CELL 19: Verify Dataset =====
print(f"✅ Dataset ready for training with {len(dataset)} examples")
print(f"📊 Final dataset features: {dataset.features}")

print("🎯 Dataset conversion complete! Ready for training.")

# ===== CELL 20: Test Chat Template =====
# Optional: Test that the chat template is working
test_convo = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]

test_text = tokenizer.apply_chat_template(test_convo, tokenize=False, add_generation_prompt=False)
print("🧪 Chat template test:")
print(test_text[:100] + "...") 