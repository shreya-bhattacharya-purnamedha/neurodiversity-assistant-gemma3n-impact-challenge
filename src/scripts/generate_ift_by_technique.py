

import json

def generate_ift_dataset_by_technique():
    # Load the regulation techniques dataset
    try:
        with open('dataset/json/2_Regulation_Techniques.json', 'r') as f:
            techniques = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {e.doc}: {e.msg}")
        return

    ift_dataset = []
    for technique in techniques:
        # Ensure the necessary keys are present
        if 'Self_Regulation_Technique_Name' in technique and 'Description' in technique and 'Instructions' in technique:
            
            technique_name = technique['Self_Regulation_Technique_Name']
            
            # Clean up the instructions, replacing <br> with newlines
            instructions = technique.get('Instructions', 'No instructions provided.')
            if isinstance(instructions, str):
                instructions = instructions.replace('<br>', '\n')

            # Construct the output with the technique details
            output_text = (
                f"**{technique['Self_Regulation_Technique_Name']}**\n\n"
                f"**What it is:** {technique['Description']}\n\n"
                f"**How to do it:**\n{instructions}"
            )

            ift_dataset.append({
                "instruction": "You are a neurodiversity assistant. A user is asking for details about a specific self-regulation technique. Provide the information clearly and concisely.",
                "input": f"Tell me more about {technique_name}.",
                "output": output_text
            })

    # Save the new dataset
    try:
        with open('dataset/json/ift_dataset_by_technique_name.json', 'w') as f:
            json.dump(ift_dataset, f, indent=2)
        print("IFT dataset generated successfully from techniques.")
        print("New file created at: dataset/json/ift_dataset_by_technique_name.json")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    generate_ift_dataset_by_technique()

