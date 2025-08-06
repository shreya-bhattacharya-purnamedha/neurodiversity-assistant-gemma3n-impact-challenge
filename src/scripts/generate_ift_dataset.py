import json

def generate_ift_dataset():
    # Load the datasets
    with open('dataset/json/1_User_Problems_and_Categories.json', 'r') as f:
        user_problems = json.load(f)
    with open('dataset/json/2_Regulation_Techniques.json', 'r') as f:
        techniques = json.load(f)
    with open('dataset/json/3_regulation_technique_to_problem_categories.json', 'r') as f:
        technique_mappings = json.load(f)

    # Create a dictionary for quick lookup of techniques by name
    techniques_dict = {tech['Self_Regulation_Technique_Name']: tech for tech in techniques}

    # Create a dictionary to map problem categories to techniques
    problem_to_techniques = {}
    for mapping in technique_mappings:
        problem_category = mapping['User_Problem_Category']
        if problem_category not in problem_to_techniques:
            problem_to_techniques[problem_category] = []
        problem_to_techniques[problem_category].append(mapping['Self_Regulation_Technique_Name'])

    # Generate the IFT dataset
    ift_dataset = []
    for problem in user_problems:
        user_query = problem['Sample_User_Query']
        problem_category = problem['User_Problem_Category']

        if problem_category in problem_to_techniques:
            for tech_name in problem_to_techniques[problem_category]:
                if tech_name in techniques_dict:
                    technique = techniques_dict[tech_name]
                    
                    # Clean up the instructions
                    instructions = technique.get('Instructions', 'No instructions provided.')
                    if isinstance(instructions, str):
                        instructions = instructions.replace('<br>', '\n')

                    output = (
                        f"Based on what you're feeling, here is a technique that might help:\n\n"
                        f"**{technique['Self_Regulation_Technique_Name']}**\n\n"
                        f"**What it is:** {technique['Description']}\n\n"
                        f"**How to do it:**\n{instructions}"
                    )

                    ift_dataset.append({
                        "instruction": "You are a neurodiversity assistant. A user comes to you with the following problem. Your job is to suggest a relevant self-regulation technique.",
                        "input": user_query,
                        "output": output
                    })

    # Save the new dataset
    with open('dataset/json/ift_dataset.json', 'w') as f:
        json.dump(ift_dataset, f, indent=2)

    print("IFT dataset generated successfully and saved to dataset/json/ift_dataset.json")

if __name__ == '__main__':
    generate_ift_dataset()
