import json

# Load both JSON files
with open('output1.json', 'r') as file:
    user_json = json.load(file)

with open('output.json', 'r') as file:
    improved_json = json.load(file)

# Function to summarize JSON data for comparison
def summarize_json(json_data):
    summary = {
        "vertices_count": len(json_data.get("vertices", {})),
        "edges_count": len(json_data.get("edges", {})),
        "faces_count": len(json_data.get("faces", {})),
        "planes_count": len(json_data.get("surfaces", {}).get("planes", [])),
        "cylinders_count": len(json_data.get("surfaces", {}).get("cylinders", []))
    }
    return summary

# Summarize both JSON files
user_json_summary = summarize_json(user_json)
improved_json_summary = summarize_json(improved_json)

# Print summaries
print("User JSON Summary:", user_json_summary)
print("Improved JSON Summary:", improved_json_summary)
