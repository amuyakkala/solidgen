# import json
# import matplotlib.pyplot as plt

# def visualize_brep(json_path, image_path=None):
#     # Load the JSON data
#     with open(json_path, 'r') as file:
#         data = json.load(file)

#     # Extract data from the JSON
#     vertices = data.get('vertices', {})
#     edges = data.get('edges', {})
#     faces = data.get('faces', {})
#     planes = data.get('surfaces', {}).get('planes', [])
#     cylinders = data.get('surfaces', {}).get('cylinders', [])

#     # Create a plot
#     plt.figure(figsize=(10, 10))
#     ax = plt.gca()

#     # If an image path is provided, display the image as a background
#     if image_path:
#         image = plt.imread(image_path)
#         ax.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])

#     # Plot vertices
#     for key, value in vertices.items():
#         x, y, _ = value
#         ax.plot(x, y, 'ro')  # Red dot for vertices
#         ax.text(x, y, key, color='yellow', fontsize=8)

#     # Plot edges
#     for key, value in edges.items():
#         v1, v2 = value
#         x1, y1, _ = vertices[v1]
#         x2, y2, _ = vertices[v2]
#         ax.plot([x1, x2], [y1, y2], 'g-')  # Green line for edges

#     # Optionally, plot faces, planes, and cylinders
#     for plane in planes:
#         points = plane['points']
#         x = [p[0] for p in points]
#         y = [p[1] for p in points]
#         ax.fill(x, y, alpha=0.3)

#     for cylinder in cylinders:
#         arc1 = cylinder['arc1']
#         arc2 = cylinder['arc2']
#         center1 = arc1['center']
#         center2 = arc2['center']
#         radius1 = arc1['radius']
#         radius2 = arc2['radius']
#         ax.add_patch(plt.Circle(center1, radius1, color='b', fill=False))
#         ax.add_patch(plt.Circle(center2, radius2, color='b', fill=False))
#         ax.plot([center1[0], center2[0]], [center1[1], center2[1]], 'b-')

#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('BREP Visualization')
#     plt.show()

# # Usage example
# json_path = 'output1.json'  # Replace with your JSON output path
# image_path = './images/image1.jpeg'  # Replace with your image path if needed

# visualize_brep(json_path, image_path)



import json

def print_json_details(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    print("Vertices:")
    for key, value in data.get('vertices', {}).items():
        print(f"{key}: {value}")
    
    print("\nEdges:")
    for key, value in data.get('edges', {}).items():
        print(f"{key}: {value}")
    
    print("\nPlanes:")
    for plane in data.get('surfaces', {}).get('planes', []):
        print(plane)
    
    print("\nCylinders:")
    for cylinder in data.get('surfaces', {}).get('cylinders', []):
        print(cylinder)

# Usage example
json_path = 'output1.json'  # Replace with your JSON output path
print_json_details(json_path)