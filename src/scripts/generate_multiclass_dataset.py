
import json

def create_multiclass_classification_dataset():
    # Load the original dataset
    try:
        with open('dataset/json/1_User_Problems_and_Categories.json', 'r') as f:
            user_problems = json.load(f)
    except FileNotFoundError:
        print("Error: '1_User_Problems_and_Categories.json' not found in 'dataset/json/'.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from '1_User_Problems_and_Categories.json'.")
        return

    multiclass_dataset = []
    for item in user_problems:
        # Ensure all required keys are present
        if all(key in item for key in ['Sample_User_Query', 'User_Energy_States', 'User_Problem_Category', 'User_Cognitive_Energy_Levels']):
            text = item['Sample_User_Query']
            labels = [
                item['User_Energy_States'],
                item['User_Problem_Category'],
                item['User_Cognitive_Energy_Levels']
            ]
            
            multiclass_dataset.append({
                "text": text,
                "labels": labels
            })

    # Save the new dataset
    try:
        with open('dataset/json/multiclass_classification_dataset.json', 'w') as f:
            json.dump(multiclass_dataset, f, indent=2)
        print("Successfully converted to multi-class classification format.")
        print("New file created at: dataset/json/multiclass_classification_dataset.json")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    create_multiclass_classification_dataset()
