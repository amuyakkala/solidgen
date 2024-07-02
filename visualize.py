import json
import matplotlib.pyplot as plt
import cv2

def visualize_json(image_path, json_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    ax = plt.gca()

    # Plot vertices
    vertices = data.get('vertices', {})
    for key, value in vertices.items():
        x, y, _ = value
        ax.plot(x, y, 'ro')  # Red dot for vertices
        ax.text(x, y, key, color='yellow', fontsize=8)

    # Plot edges
    edges = data.get('edges', {})
    for key, value in edges.items():
        v1, v2 = value
        x1, y1, _ = vertices[v1]
        x2, y2, _ = vertices[v2]
        ax.plot([x1, x2], [y1, y2], 'g-')  # Green line for edges

    # Optionally, plot planes and cylinders
    # ...

    plt.show()

# Usage example
image_path = './images/image1.jpeg'  # Replace with your image path
json_path = 'output1.json'  # Replace with your JSON output path

visualize_json(image_path, json_path)
