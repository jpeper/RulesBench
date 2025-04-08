import os
import json

# Load the source JSON data
with open("test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

# Output folder
output_dir = "test_cases"
os.makedirs(output_dir, exist_ok=True)

# Helper to sanitize file IDs
def sanitize_id(id_str):
    return str(id_str).strip().zfill(3) if str(id_str).isdigit() else f"unknown_{id_str}"

# Iterate through the test cases and write each as a .json file
for case in test_cases:
    file_id = sanitize_id(case["ID"])
    filename = f"test_{file_id}.json"
    filepath = os.path.join(output_dir, filename)

    # Add the new fields
    case["game_state_json"] = {
        "map": {},
        "pieces": [],
        "player_status": {},
        "notes": "Placeholder for structured game state"
    }
    case["game_state_url"] = f"https://example.com/game_state_images/test_{file_id}.png"

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2)