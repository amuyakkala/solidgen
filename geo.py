




# # import cv2
# # import numpy as np
# # import os
# # import matplotlib.pyplot as plt

# # from sklearn.cluster import DBSCAN

# # def preprocess_image(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# #     return blurred

# # def detect_edges(image):
# #     edges = cv2.Canny(image, 50, 150)
# #     return edges

# # def detect_planar_surfaces(image, edges):
# #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     planar_surfaces = []
# #     for contour in contours:
# #         epsilon = 0.02 * cv2.arcLength(contour, True)
# #         approx = cv2.approxPolyDP(contour, epsilon, True)
# #         if len(approx) == 4:  # Assuming rectangular surfaces
# #             planar_surfaces.append(approx)
# #     return planar_surfaces

# # def detect_cylindrical_surfaces(image, edges):
# #     circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
# #                                param1=50, param2=30, minRadius=0, maxRadius=0)
# #     if circles is not None:
# #         circles = np.uint16(np.around(circles))
# #         return circles[0, :]
# #     return []

# # def detect_line_segments(edges):
# #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
# #     if lines is not None:
# #         return [line[0] for line in lines]
# #     return []

# # def detect_arcs(edges):
# #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     arcs = []
# #     for contour in contours:
# #         epsilon = 0.01 * cv2.arcLength(contour, True)
# #         approx = cv2.approxPolyDP(contour, epsilon, True)
# #         if len(approx) > 4 and len(approx) < 10:  # Potential arc
# #             ellipse = cv2.fitEllipse(contour)
# #             arcs.append(ellipse)
# #     return arcs

# # def extract_geometry(image_path):
# #     image = cv2.imread(image_path)
# #     if image is None:
# #         raise FileNotFoundError(f"Error: Image not found or cannot be read at {image_path}")
    
# #     preprocessed = preprocess_image(image)
# #     edges = detect_edges(preprocessed)
    
# #     planar_surfaces = detect_planar_surfaces(preprocessed, edges)
# #     cylindrical_surfaces = detect_cylindrical_surfaces(preprocessed, edges)
# #     line_segments = detect_line_segments(edges)
# #     arcs = detect_arcs(edges)
    
# #     return {
# #         'planar': planar_surfaces,
# #         'cylindrical': cylindrical_surfaces,
# #         'lines': line_segments,
# #         'arcs': arcs
# #     }, image

# # def validate_geometry(extracted_geometry, ground_truth):
# #     # This is a placeholder function. You'll need to implement a proper
# #     # validation method based on your specific requirements and ground truth format.
# #     total_extracted = (len(extracted_geometry['planar']) +
# #                        len(extracted_geometry['cylindrical']) +
# #                        len(extracted_geometry['lines']) +
# #                        len(extracted_geometry['arcs']))
# #     total_ground_truth = sum(len(gt) for gt in ground_truth.values())
# #     return total_extracted / total_ground_truth if total_ground_truth > 0 else 0

# # def visualize_results(image, geometry):
# #     vis_image = image.copy()
    
# #     # Visualize planar surfaces
# #     cv2.drawContours(vis_image, geometry['planar'], -1, (0, 255, 0), 2)
    
# #     # Visualize cylindrical surfaces
# #     for (x, y, r) in geometry['cylindrical']:
# #         cv2.circle(vis_image, (x, y), r, (255, 0, 0), 2)
    
# #     # Visualize line segments
# #     for line in geometry['lines']:
# #         x1, y1, x2, y2 = line
# #         cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
# #     # Visualize arcs
# #     for arc in geometry['arcs']:
# #         cv2.ellipse(vis_image, arc, (255, 255, 0), 2)
    
# #     plt.figure(figsize=(12, 8))
# #     plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
# #     plt.title("Extracted Geometry")
# #     plt.axis('off')
# #     plt.show()

# # def main():
# #     image_folder = "images"
# #     ground_truth_file = "ground_truth.txt"
    
# #     # Load ground truth data (you'll need to create this file)
# #     ground_truth_data = {}
# #     if os.path.exists(ground_truth_file):
# #         with open(ground_truth_file, 'r') as f:
# #             ground_truth_data = eval(f.read())
# #     else:
# #         print(f"Warning: Ground truth file '{ground_truth_file}' not found.")
    
# #     total_accuracy = 0
# #     processed_images = 0
    
# #     for image_file in os.listdir(image_folder):
# #         if image_file.endswith(('.jpg', '.png', '.jpeg')):
# #             image_path = os.path.join(image_folder, image_file)
# #             try:
# #                 extracted_geometry, image = extract_geometry(image_path)
                
# #                 ground_truth = ground_truth_data.get(image_file, {})
# #                 accuracy = validate_geometry(extracted_geometry, ground_truth)
# #                 total_accuracy += accuracy
# #                 processed_images += 1
                
# #                 print(f"Processed {image_file} - Accuracy: {accuracy:.2f}")
# #                 print(f"Planar surfaces: {len(extracted_geometry['planar'])}")
# #                 print(f"Cylindrical surfaces: {len(extracted_geometry['cylindrical'])}")
# #                 print(f"Line segments: {len(extracted_geometry['lines'])}")
# #                 print(f"Arcs: {len(extracted_geometry['arcs'])}")
                
# #                 visualize_results(image, extracted_geometry)
            
# #             except Exception as e:
# #                 print(f"Error processing {image_file}: {str(e)}")
    
# #     if processed_images > 0:
# #         average_accuracy = total_accuracy / processed_images
# #         print(f"\nAverage accuracy across {processed_images} images: {average_accuracy:.2f}")
# #     else:
# #         print("No images were processed.")

# # if __name__ == "__main__":
# #     main()










# # import cv2
# # import numpy as np

# # # Load the image
# # image_path = './images/image1.jpeg'
# # image = cv2.imread(image_path)

# # # Check if the image was loaded successfully
# # if image is None:
# #     print("Error: Unable to load the image. Please check the file path.")
# # else:
# #     # Convert the image to grayscale
# #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #     # Apply Canny edge detection
# #     edges = cv2.Canny(gray_image, 50, 150)

# #     # Find contours in the edge image
# #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     # Initialize lists to store vertices, edges, and faces
# #     vertices = []
# #     edges = []
# #     faces = []

# #     # Extract vertices and edges from contours
# #     vertex_id = 1
# #     edge_id = 1
# #     for contour in contours:
# #         area = cv2.contourArea(contour)
# #         perimeter = cv2.arcLength(contour, True)
        
# #         # Check if the perimeter is not zero before calculating circularity
# #         if perimeter != 0:
# #             circularity = 4 * np.pi * area / (perimeter * perimeter)
# #         else:
# #             circularity = 0  # Set circularity to 0 if perimeter is zero

# #         # Extract vertices
# #         for point in contour:
# #             x, y = point[0]
# #             vertices.append({"id": vertex_id, "x": x, "y": y, "z": 0})
# #             vertex_id += 1

# #         # Extract edges
# #         for i in range(len(contour) - 1):
# #             start_vertex_id = vertex_id - len(contour) + i
# #             end_vertex_id = vertex_id - len(contour) + i + 1
# #             edges.append({"id": edge_id, "start_vertex_id": start_vertex_id, "end_vertex_id": end_vertex_id})
# #             edge_id += 1

# #     # Create a face using all the extracted edges
# #     faces.append({"id": 1, "edge_ids": [edge["id"] for edge in edges]})

# #     # Construct the indexed B-rep format
# #     indexed_brep = {
# #         "vertices": vertices,
# #         "edges": edges,
# #         "faces": faces
# #     }

# #     # Print or save the indexed B-rep format
# #     print(indexed_brep)




# #main code 
# # import cv2
# # import numpy as np
# # from sklearn.cluster import DBSCAN

# # def preprocess_image(image_path):
# #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     if img is None:
# #         raise ValueError(f"Error: Image not found or cannot be read at {image_path}")
# #     img = cv2.GaussianBlur(img, (5, 5), 0)
# #     return img

# # def detect_edges(img):
# #     edges = cv2.Canny(img, 50, 150)
# #     return edges

# # def detect_lines(edges):
# #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
# #     return lines

# # def detect_circles(img):
# #     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
# #     return circles

# # def cluster_line_endpoints(lines):
# #     if lines is None or len(lines) == 0:
# #         return []

# #     endpoints = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])

# #     if endpoints.ndim != 2:
# #         raise ValueError("endpoints must be a 2-dimensional array")

# #     clustering = DBSCAN(eps=5, min_samples=2).fit(endpoints)
# #     unique_labels = np.unique(clustering.labels_)
# #     clustered_points = [endpoints[clustering.labels_ == label] for label in unique_labels if label != -1]
# #     return clustered_points

# # def average_cluster_points(clustered_points):
# #     return [np.mean(cluster, axis=0) for cluster in clustered_points]

# # def connect_line_segments(averaged_points, lines):
# #     connected_lines = []
# #     for line in lines[:, 0]:
# #         start = line[:2]
# #         end = line[2:]
# #         start_idx = np.argmin(np.linalg.norm(averaged_points - start, axis=1))
# #         end_idx = np.argmin(np.linalg.norm(averaged_points - end, axis=1))
# #         if start_idx != end_idx:
# #             connected_lines.append((start_idx, end_idx))
# #     return connected_lines

# # def extract_geometry(image_path):
# #     img = preprocess_image(image_path)
# #     edges = detect_edges(img)
# #     lines = detect_lines(edges)
# #     circles = detect_circles(img)

# #     if lines is not None and len(lines) > 0:
# #         clustered_points = cluster_line_endpoints(lines)
# #         averaged_points = average_cluster_points(clustered_points)
# #         connected_lines = connect_line_segments(averaged_points, lines)
# #     else:
# #         averaged_points = []
# #         connected_lines = []

# #     return averaged_points, connected_lines, circles

# # def generate_solidgen_output(points, lines, circles):
# #     solidgen_output = {
# #         'vertices': [],
# #         'edges': [],
# #         'faces': []
# #     }

# #     # Add vertices
# #     for point in points:
# #         solidgen_output['vertices'].append((int(point[0]), int(point[1])))  # Assuming z-coordinate is 0

# #     # Add edges
# #     for start, end in lines:
# #         solidgen_output['edges'].append((int(start), int(end)))

# #     # Add faces (assuming circles represent faces)
# #     if circles is not None:
# #         circles = np.round(circles[0, :]).astype(int)
# #         for i, (x, y, r) in enumerate(circles):
# #             solidgen_output['faces'].append((0, 1, 2))  # Placeholder, replace with actual logic

# #     return solidgen_output

# # def main(image_path, output_file="solidgen_output.txt"):
# #     try:
# #         points, lines, circles = extract_geometry(image_path)
# #         solidgen_output = generate_solidgen_output(points, lines, circles)
        
# #         # Print or save the output
# #         print("Vertices:", solidgen_output['vertices'])
# #         print("Edges:", solidgen_output['edges'])
# #         print("Faces:", solidgen_output['faces'])

# #     except ValueError as e:
# #         print(e)

# # if __name__ == "__main__":
# #     image_path = "./images/image1.jpeg"  # Ensure this path is correct
# #     main(image_path)





# 
# second main code which give soldgen fromat in terminal
# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Error: Image not found or cannot be read at {image_path}")
#     image = cv2.GaussianBlur(image, (5, 5), 0)
#     return image

# def detect_edges(image):
#     edges = cv2.Canny(image, 50, 150)
#     return edges

# def detect_lines_and_arcs(edges):
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
#     circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)
    
#     line_segments = []
#     arcs = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             line_segments.append(((x1, y1), (x2, y2)))
    
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for circle in circles[0, :]:
#             center = (circle[0], circle[1])
#             radius = circle[2]
#             arcs.append((center, radius))
    
#     return line_segments, arcs

# def detect_planes(line_segments):
#     if not line_segments:
#         return []
    
#     points = np.array(line_segments).reshape(-1, 2)
#     clustering = DBSCAN(eps=10, min_samples=3).fit(points)
    
#     planes = []
#     for label in set(clustering.labels_):
#         if label != -1:
#             plane_points = points[clustering.labels_ == label]
#             planes.append(plane_points)
    
#     return planes

# def detect_cylinders(arcs):
#     # In a 2D image, cylinders will appear as parallel lines or arcs
#     # This is a simplified approach and may need refinement
#     if len(arcs) < 2:
#         return []
    
#     cylinders = []
#     for i in range(len(arcs)):
#         for j in range(i+1, len(arcs)):
#             if np.isclose(arcs[i][1], arcs[j][1], rtol=0.1):  # Compare radii
#                 cylinders.append((arcs[i], arcs[j]))
    
#     return cylinders

# def convert_to_solidgen_format(line_segments, arcs, planes, cylinders):
#     solidgen_data = {
#         "vertices": {},
#         "edges": {},
#         "faces": {},
#         "surfaces": {
#             "planes": [],
#             "cylinders": []
#         }
#     }
    
#     vertex_id = 1
#     edge_id = 1
#     face_id = 1
    
#     # Add line segments
#     for i, (start, end) in enumerate(line_segments):
#         solidgen_data["vertices"][f"v{vertex_id}"] = list(start) + [0]
#         solidgen_data["vertices"][f"v{vertex_id+1}"] = list(end) + [0]
#         solidgen_data["edges"][f"e{edge_id}"] = [f"v{vertex_id}", f"v{vertex_id+1}"]
#         vertex_id += 2
#         edge_id += 1
    
#     # Add arcs
#     for i, (center, radius) in enumerate(arcs):
#         # Approximate arc with 3 points
#         angles = [0, np.pi/2, np.pi]
#         arc_vertices = []
#         for angle in angles:
#             x = center[0] + radius * np.cos(angle)
#             y = center[1] + radius * np.sin(angle)
#             solidgen_data["vertices"][f"v{vertex_id}"] = [int(x), int(y), 0]
#             arc_vertices.append(f"v{vertex_id}")
#             vertex_id += 1
        
#         solidgen_data["edges"][f"e{edge_id}"] = [arc_vertices[0], arc_vertices[1]]
#         solidgen_data["edges"][f"e{edge_id+1}"] = [arc_vertices[1], arc_vertices[2]]
#         edge_id += 2
        
#         solidgen_data["faces"][f"f{face_id}"] = [f"e{edge_id-2}", f"e{edge_id-1}"]
#         face_id += 1
    
#     # Add planes
#     for i, plane in enumerate(planes):
#         solidgen_data["surfaces"]["planes"].append({
#             "id": f"plane{i+1}",
#             "points": plane.tolist()
#         })
    
#     # Add cylinders
#     for i, (arc1, arc2) in enumerate(cylinders):
#         solidgen_data["surfaces"]["cylinders"].append({
#             "id": f"cylinder{i+1}",
#             "arc1": {"center": arc1[0], "radius": arc1[1]},
#             "arc2": {"center": arc2[0], "radius": arc2[1]}
#         })
    
#     return solidgen_data

# def extract_geometry(image_path):
#     image = preprocess_image(image_path)
#     edges = detect_edges(image)
#     line_segments, arcs = detect_lines_and_arcs(edges)
#     planes = detect_planes(line_segments)
#     cylinders = detect_cylinders(arcs)
    
#     solidgen_data = convert_to_solidgen_format(line_segments, arcs, planes, cylinders)
#     return solidgen_data

# # Sample usage
# if __name__ == "__main__":
#     image_path = './images/image1.jpeg'  # Replace with your image path
#     try:
#         solidgen_data = extract_geometry(image_path)
#         print("Extracted Geometry in SolidGen Format:")
#         print(solidgen_data)
#     except Exception as e:
#         print(f"Error: {str(e)}")




# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN
# import json

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Error: Image not found or cannot be read at {image_path}")
#     image = cv2.GaussianBlur(image, (5, 5), 0)
#     return image

# def detect_edges(image):
#     edges = cv2.Canny(image, 50, 150)
#     return edges

# def detect_lines_and_arcs(edges):
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
#     circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)
    
#     line_segments = []
#     arcs = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             line_segments.append(((int(x1), int(y1)), (int(x2), int(y2))))
    
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for circle in circles[0, :]:
#             center = (int(circle[0]), int(circle[1]))
#             radius = int(circle[2])
#             arcs.append((center, radius))
    
#     return line_segments, arcs

# def detect_planes(line_segments):
#     if not line_segments:
#         return []
    
#     points = np.array(line_segments).reshape(-1, 2)
#     clustering = DBSCAN(eps=10, min_samples=3).fit(points)
    
#     planes = []
#     for label in set(clustering.labels_):
#         if label != -1:
#             plane_points = points[clustering.labels_ == label]
#             planes.append(plane_points.tolist())
    
#     return planes

# def detect_cylinders(arcs):
#     if len(arcs) < 2:
#         return []
    
#     cylinders = []
#     for i in range(len(arcs)):
#         for j in range(i+1, len(arcs)):
#             if np.isclose(arcs[i][1], arcs[j][1], rtol=0.1):
#                 cylinders.append((arcs[i], arcs[j]))
    
#     return cylinders

# def convert_to_solidgen_format(line_segments, arcs, planes, cylinders):
#     solidgen_data = {
#         "vertices": {},
#         "edges": {},
#         "faces": {},
#         "surfaces": {
#             "planes": [],
#             "cylinders": []
#         }
#     }
    
#     vertex_id = 1
#     edge_id = 1
#     face_id = 1
    
#     # Add line segments
#     for i, (start, end) in enumerate(line_segments):
#         solidgen_data["vertices"][f"v{vertex_id}"] = list(start) + [0]
#         solidgen_data["vertices"][f"v{vertex_id+1}"] = list(end) + [0]
#         solidgen_data["edges"][f"e{edge_id}"] = [f"v{vertex_id}", f"v{vertex_id+1}"]
#         vertex_id += 2
#         edge_id += 1
    
#     # Add arcs
#     for i, (center, radius) in enumerate(arcs):
#         angles = [0, np.pi/2, np.pi]
#         arc_vertices = []
#         for angle in angles:
#             x = int(center[0] + radius * np.cos(angle))
#             y = int(center[1] + radius * np.sin(angle))
#             solidgen_data["vertices"][f"v{vertex_id}"] = [x, y, 0]
#             arc_vertices.append(f"v{vertex_id}")
#             vertex_id += 1
        
#         solidgen_data["edges"][f"e{edge_id}"] = [arc_vertices[0], arc_vertices[1]]
#         solidgen_data["edges"][f"e{edge_id+1}"] = [arc_vertices[1], arc_vertices[2]]
#         edge_id += 2
        
#         solidgen_data["faces"][f"f{face_id}"] = [f"e{edge_id-2}", f"e{edge_id-1}"]
#         face_id += 1
    
#     # Add planes
#     for i, plane in enumerate(planes):
#         solidgen_data["surfaces"]["planes"].append({
#             "id": f"plane{i+1}",
#             "points": plane
#         })
    
#     # Add cylinders
#     for i, (arc1, arc2) in enumerate(cylinders):
#         solidgen_data["surfaces"]["cylinders"].append({
#             "id": f"cylinder{i+1}",
#             "arc1": {"center": list(arc1[0]), "radius": arc1[1]},
#             "arc2": {"center": list(arc2[0]), "radius": arc2[1]}
#         })
    
#     return solidgen_data

# def extract_geometry(image_path, output_path):
#     image = preprocess_image(image_path)
#     edges = detect_edges(image)
#     line_segments, arcs = detect_lines_and_arcs(edges)
#     planes = detect_planes(line_segments)
#     cylinders = detect_cylinders(arcs)
    
#     solidgen_data = convert_to_solidgen_format(line_segments, arcs, planes, cylinders)
    
#     with open(output_path, 'w') as f:
#         json.dump(solidgen_data, f, indent=2)
    
#     print(f"Extracted Geometry in SolidGen Format written to: {output_path}")

# # Sample usage
# if __name__ == "__main__":
#     image_path = './images/image1.jpeg'  # Replace with your image path
#     output_path = './output1.json'  # Replace with your desired output path
#     try:
#         extract_geometry(image_path, output_path)
#     except Exception as e:
#         print(f"Error: {str(e)}")




import cv2
from sklearn.cluster import DBSCAN

import numpy as np
import json

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Image not found or cannot be read at {image_path}")
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def detect_edges(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def detect_lines_and_arcs(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)
    
    line_segments = []
    arcs = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append(((int(x1), int(y1)), (int(x2), int(y2))))
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            arcs.append((center, radius))
    
    return line_segments, arcs

def detect_planes(line_segments):
    if not line_segments:
        return []
    
    points = np.array(line_segments).reshape(-1, 2)
    clustering = DBSCAN(eps=10, min_samples=3).fit(points)
    
    planes = []
    for label in set(clustering.labels_):
        if label != -1:
            plane_points = points[clustering.labels_ == label]
            planes.append(plane_points.tolist())
    
    return planes

def detect_cylinders(arcs):
    if len(arcs) < 2:
        return []
    
    cylinders = []
    for i in range(len(arcs)):
        for j in range(i+1, len(arcs)):
            if np.isclose(arcs[i][1], arcs[j][1], rtol=0.1):
                cylinders.append((arcs[i], arcs[j]))
    
    return cylinders

def convert_to_solidgen_format(line_segments, arcs, planes, cylinders):
    solidgen_data = {
        "vertices": {},
        "edges": {},
        "faces": {},
        "surfaces": {
            "planes": [],
            "cylinders": []
        }
    }
    
    vertex_id = 1
    edge_id = 1
    face_id = 1
    
    # Add line segments
    for i, (start, end) in enumerate(line_segments):
        solidgen_data["vertices"][f"v{vertex_id}"] = list(start) + [0]
        solidgen_data["vertices"][f"v{vertex_id+1}"] = list(end) + [0]
        solidgen_data["edges"][f"e{edge_id}"] = [f"v{vertex_id}", f"v{vertex_id+1}"]
        vertex_id += 2
        edge_id += 1
    
    # Add arcs
    for i, (center, radius) in enumerate(arcs):
        angles = [0, np.pi/2, np.pi]
        arc_vertices = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            solidgen_data["vertices"][f"v{vertex_id}"] = [x, y, 0]
            arc_vertices.append(f"v{vertex_id}")
            vertex_id += 1
        
        solidgen_data["edges"][f"e{edge_id}"] = [arc_vertices[0], arc_vertices[1]]
        solidgen_data["edges"][f"e{edge_id+1}"] = [arc_vertices[1], arc_vertices[2]]
        edge_id += 2
        
        solidgen_data["faces"][f"f{face_id}"] = [f"e{edge_id-2}", f"e{edge_id-1}"]
        face_id += 1
    
    # Add planes
    for i, plane in enumerate(planes):
        solidgen_data["surfaces"]["planes"].append({
            "id": f"plane{i+1}",
            "points": plane
        })
    
    # Add cylinders
    for i, (arc1, arc2) in enumerate(cylinders):
        solidgen_data["surfaces"]["cylinders"].append({
            "id": f"cylinder{i+1}",
            "arc1": {"center": list(arc1[0]), "radius": arc1[1]},
            "arc2": {"center": list(arc2[0]), "radius": arc2[1]}
        })
    
    return solidgen_data

def extract_geometry(image_path, output_path):
    image = preprocess_image(image_path)
    edges = detect_edges(image)
    line_segments, arcs = detect_lines_and_arcs(edges)
    planes = detect_planes(line_segments)
    cylinders = detect_cylinders(arcs)
    
    solidgen_data = convert_to_solidgen_format(line_segments, arcs, planes, cylinders)
    
    with open(output_path, 'w') as f:
        json.dump(solidgen_data, f, indent=2)
    
    print(f"Extracted Geometry in SolidGen Format written to: {output_path}")

# Sample usage
if __name__ == "__main__":
    image_path = './images/image1.jpeg'  # Replace with your image path
    output_path = './output.json'  # Replace with your desired output path
    try:
        extract_geometry(image_path, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
