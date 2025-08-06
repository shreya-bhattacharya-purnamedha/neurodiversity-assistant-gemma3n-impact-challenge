

import json
import random

def create_reasoning_ift_dataset():
    # Define file paths
    path_multiclass = 'dataset/ift/1_ift_multiclass_classification_dataset.json'
    path_by_category = 'dataset/ift/2_ift_dataset_by_category.json'
    path_by_technique = 'dataset/ift/3_ift_dataset_by_technique_name.json'
    output_path = 'dataset/ift/reasoning_and_techniques_ift_dataset.json'

    # Load the datasets
    try:
        with open(path_multiclass, 'r') as f:
            multiclass_data = json.load(f)
        with open(path_by_category, 'r') as f:
            by_category_data = json.load(f)
        with open(path_by_technique, 'r') as f:
            by_technique_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {e.doc}: {e.msg}")
        return

    # --- Create helper dictionaries for efficient lookups ---

    # 1. Dictionary for technique details
    technique_details = {}
    for item in by_technique_data:
        # Extract the technique name from the input string
        name = item['input'].replace('Tell me more about ', '').strip('.')
        technique_details[name] = item['output']

    # 2. Dictionary to map categories to techniques
    category_to_techniques = {}
    for item in by_category_data:
        # Extract categories from the input string
        lines = item['input'].strip().split('\n')
        try:
            energy_state = lines[1].split(': ')[1]
            problem_category = lines[2].split(': ')[1]
            cognitive_level = lines[3].split(': ')[1]
            # Extract technique name from the output
            tech_name = item['output'].split('\n\n')[0].replace('**', '')

            key = (energy_state, problem_category, cognitive_level)
            if key not in category_to_techniques:
                category_to_techniques[key] = []
            category_to_techniques[key].append(tech_name)
        except IndexError:
            # Skip malformed entries
            continue

    # --- Generate the final IFT dataset ---
    final_ift_dataset = []
    for item in multiclass_data:
        user_input = item['input']
        classification_output = item['output']

        # Extract the categories from the classification output
        try:
            lines = classification_output.strip().split('\n')
            energy_state = lines[0].split(': ')[1]
            problem_category = lines[1].split(': ')[1].lstrip('- ')
            cognitive_level = lines[2].split(': ')[1].lstrip('- ')
        except IndexError:
            continue

        # Reasoning step
        reasoning = (
            f"Okay, it sounds like you're dealing with a few things:\n"
            f"{classification_output}\n\n"
            f"Based on this, here are a few techniques that could be particularly helpful right now:"
        )

        # Find matching techniques
        lookup_key = (energy_state, problem_category, cognitive_level)
        matching_tech_names = category_to_techniques.get(lookup_key, [])

        # Get details for the top 5 techniques
        top_techniques = []
        if matching_tech_names:
            # Shuffle to get a random top 5 if more than 5 exist
            random.shuffle(matching_tech_names)
            for tech_name in matching_tech_names[:5]:
                if tech_name in technique_details:
                    top_techniques.append(technique_details[tech_name])
        
        if top_techniques:
            final_output = reasoning + "\n\n" + "\n\n---\n\n".join(top_techniques)

            final_ift_dataset.append({
                "instruction": "You are a helpful neurodiversity assistant. First, reason about the user's input by classifying their state. Then, based on that reasoning, provide up to 5 relevant self-regulation techniques.",
                "input": user_input,
                "output": final_output
            })

    # Save the new dataset
    try:
        with open(output_path, 'w') as f:
            json.dump(final_ift_dataset, f, indent=2)
        print(f"Successfully created the new reasoning-based IFT dataset.")
        print(f"New file available at: {output_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    create_reasoning_ift_dataset()
