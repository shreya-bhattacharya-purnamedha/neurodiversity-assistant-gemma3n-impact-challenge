

import json

def generate_ift_dataset_by_category():
    # Load the necessary datasets
    try:
        with open('dataset/json/2_Regulation_Techniques.json', 'r') as f:
            techniques = json.load(f)
        with open('dataset/json/3_regulation_technique_to_problem_categories.json', 'r') as f:
            technique_mappings = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {e.doc}: {e.msg}")
        return

    # Create a dictionary for quick lookup of technique details
    techniques_dict = {tech['Self_Regulation_Technique_Name']: tech for tech in techniques}

    ift_dataset = []
    for mapping in technique_mappings:
        # Ensure all required keys are present in the mapping
        if all(key in mapping for key in ['User_Energy_States', 'User_Problem_Category', 'User_Cognitive_Energy_Levels', 'Self_Regulation_Technique_Name']):
            
            # Construct the input from the categories
            input_text = (
                f"The user is experiencing the following state:\n"
                f"- Energy State: {mapping['User_Energy_States']}\n"
                f"- Problem Category: {mapping['User_Problem_Category']}\n"
                f"- Cognitive Energy Level: {mapping['User_Cognitive_Energy_Levels']}"
            )

            tech_name = mapping['Self_Regulation_Technique_Name']
            if tech_name in techniques_dict:
                technique = techniques_dict[tech_name]
                
                # Clean up the instructions
                instructions = technique.get('Instructions', 'No instructions provided.')
                if isinstance(instructions, str):
                    instructions = instructions.replace('<br>', '\n')

                # Construct the output with the technique details
                output_text = (
                    f"Based on the user's state, here is a suggested technique:\n\n"
                    f"**{technique['Self_Regulation_Technique_Name']}**\n\n"
                    f"**What it is:** {technique['Description']}\n\n"
                    f"**How to do it:**\n{instructions}"
                )

                ift_dataset.append({
                    "instruction": "You are a neurodiversity assistant. Based on the user's categorized state, provide a relevant self-regulation technique.",
                    "input": input_text,
                    "output": output_text
                })

    # Save the new dataset
    try:
        with open('dataset/json/ift_dataset_by_category.json', 'w') as f:
            json.dump(ift_dataset, f, indent=2)
        print("IFT dataset generated successfully from categories.")
        print("New file created at: dataset/json/ift_dataset_by_category.json")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    generate_ift_dataset_by_category()

