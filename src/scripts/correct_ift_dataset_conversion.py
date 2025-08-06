# CORRECTED IFT DATASET CONVERSION FOR GEMMA 3N
# This fixes the instruction/input/output format to proper Gemma 3N conversational format

# ===== STEP 1: Load the IFT dataset =====
from datasets import load_dataset

# Load the dataset from the local JSON file
dataset = load_dataset("json", data_files="/content/combined.json")

# Assuming the dataset is in the 'train' split
dataset = dataset["train"]

print(f"📖 Loaded {len(dataset)} examples from combined.json")
print(f"📊 Dataset features: {dataset.features}")

# ===== STEP 2: Convert IFT format to Gemma 3N conversational format =====
def convert_ift_to_conversations(examples):
    """Convert instruction/input/output format to Gemma 3N conversations"""
    conversations = []
    
    for instruction, user_input, output in zip(examples["instruction"], examples["input"], examples["output"]):
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
                "content": [{"type": "text", "text": output}]
            }
        ]
        
        conversations.append(conversation)
    
    return {"conversations": conversations}

# Apply the conversion
dataset = dataset.map(convert_ift_to_conversations, batched=True)

print(f"✅ Converted {len(dataset)} IFT examples to conversations")
print("📝 Example conversation:")
print(dataset[0]["conversations"])

# ===== STEP 3: Apply Gemma 3N chat template =====
def formatting_prompts_func(examples):
    """Apply Gemma 3N chat template to conversations"""
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

# Apply the chat template formatting
dataset = dataset.map(formatting_prompts_func, batched=True)

print("📝 Formatted example (first 200 chars):")
print(dataset[0]["text"][:200] + "...")

# ===== STEP 4: Verify the dataset =====
print(f"✅ Dataset ready for training with {len(dataset)} examples")
print(f"📊 Final dataset features: {dataset.features}")

# ===== STEP 5: Set up chat template for tokenizer =====
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

print("🎯 Dataset conversion complete! Ready for training.") 