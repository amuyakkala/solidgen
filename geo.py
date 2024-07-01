import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def detect_planar_surfaces(image, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    planar_surfaces = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Assuming rectangular surfaces
            planar_surfaces.append(approx)
    return planar_surfaces

def detect_cylindrical_surfaces(image, edges):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    return []

def detect_line_segments(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        return [line[0] for line in lines]
    return []

def detect_arcs(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arcs = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 4 and len(approx) < 10:  # Potential arc
            ellipse = cv2.fitEllipse(contour)
            arcs.append(ellipse)
    return arcs

def extract_geometry(image_path):
    image = cv2.imread(image_path)
    preprocessed = preprocess_image(image)
    edges = detect_edges(preprocessed)
    
    planar_surfaces = detect_planar_surfaces(preprocessed, edges)
    cylindrical_surfaces = detect_cylindrical_surfaces(preprocessed, edges)
    line_segments = detect_line_segments(edges)
    arcs = detect_arcs(edges)
    
    return {
        'planar': planar_surfaces,
        'cylindrical': cylindrical_surfaces,
        'lines': line_segments,
        'arcs': arcs
    }, image

def validate_geometry(extracted_geometry, ground_truth):
    # This is a placeholder function. You'll need to implement a proper
    # validation method based on your specific requirements and ground truth format.
    total_extracted = (len(extracted_geometry['planar']) +
                       len(extracted_geometry['cylindrical']) +
                       len(extracted_geometry['lines']) +
                       len(extracted_geometry['arcs']))
    total_ground_truth = sum(len(gt) for gt in ground_truth.values())
    return total_extracted / total_ground_truth if total_ground_truth > 0 else 0

def visualize_results(image, geometry):
    vis_image = image.copy()
    
    # Visualize planar surfaces
    cv2.drawContours(vis_image, geometry['planar'], -1, (0, 255, 0), 2)
    
    # Visualize cylindrical surfaces
    for (x, y, r) in geometry['cylindrical']:
        cv2.circle(vis_image, (x, y), r, (255, 0, 0), 2)
    
    # Visualize line segments
    for line in geometry['lines']:
        x1, y1, x2, y2 = line
        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Visualize arcs
    for arc in geometry['arcs']:
        cv2.ellipse(vis_image, arc, (255, 255, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Geometry")
    plt.axis('off')
    plt.show()

def main():
    image_folder = "path/to/your/image/folder"
    ground_truth_file = "path/to/your/ground_truth.txt"
    
    # Load ground truth data (you'll need to create this file)
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = eval(f.read())
    
    total_accuracy = 0
    processed_images = 0
    
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            extracted_geometry, image = extract_geometry(image_path)
            
            if extracted_geometry:
                ground_truth = ground_truth_data.get(image_file, {})
                accuracy = validate_geometry(extracted_geometry, ground_truth)
                total_accuracy += accuracy
                processed_images += 1
                
                print(f"Processed {image_file} - Accuracy: {accuracy:.2f}")
                print(f"Planar surfaces: {len(extracted_geometry['planar'])}")
                print(f"Cylindrical surfaces: {len(extracted_geometry['cylindrical'])}")
                print(f"Line segments: {len(extracted_geometry['lines'])}")
                print(f"Arcs: {len(extracted_geometry['arcs'])}")
                
                visualize_results(image, extracted_geometry)
    
    if processed_images > 0:
        average_accuracy = total_accuracy / processed_images
        print(f"\nAverage accuracy across {processed_images} images: {average_accuracy:.2f}")
    else:
        print("No images were processed.")

if __name__ == "__main__":
    main()

