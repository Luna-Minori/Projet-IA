import cv2
import numpy as np
from PIL import Image
from multiprocessing import Process

def convert_to_ppm(frame, output_image_path):
    # Convertir l'image de la caméra en PPM (format texte)
    height, width, channels = frame.shape
    with open(output_image_path, 'w') as file:
        file.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            for x in range(width):
                # Récupérer les valeurs RGB du pixel
                r, g, b = frame[y, x]
                file.write(f"{r} {g} {b} ")
            file.write("\n")
    print(f"L'image a été convertie en PPM et sauvegardée sous {output_image_path}")

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    lower_threshold, upper_threshold = calculate_dynamic_thresholds(blurred_frame)
    edges = cv2.Canny(blurred_frame, lower_threshold, upper_threshold)
    lower_threshold, upper_threshold = adjust_thresholds_based_on_edge_count(edges, lower_threshold, upper_threshold)
    edges_adjusted = cv2.Canny(blurred_frame, lower_threshold, upper_threshold)

    # Affichage dans le thread principal
    cv2.imshow('Original', frame)
    cv2.imshow('Gris', gray_frame)
    cv2.imshow('Flou Gaussien', blurred_frame)
    cv2.imshow('Contours Dynamiques Ajustés', edges_adjusted)
    
    print(f"Seuils dynamiques: {lower_threshold}, {upper_threshold}")
    print(f"Nombre de bords détectés: {np.sum(edges_adjusted > 0)}")

def calculate_dynamic_thresholds(image, lower_bound=50, upper_bound=200, target_edge_count=1000):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    mean_intensity = np.mean(image)
    std_dev_intensity = np.std(image)
    lower_threshold = max(lower_bound, int(mean_intensity - 0.5 * std_dev_intensity))
    upper_threshold = min(upper_bound, int(mean_intensity + 0.5 * std_dev_intensity))
    lower_threshold = max(0, lower_threshold)
    upper_threshold = min(255, upper_threshold)
    return lower_threshold, upper_threshold

def adjust_thresholds_based_on_edge_count(edges, lower_threshold, upper_threshold, target_edge_count=10000):
    edge_count = np.sum(edges > 0)
    if edge_count > target_edge_count:  
        lower_threshold = min(upper_threshold - 1, lower_threshold + 10)
        upper_threshold = min(255, upper_threshold + 10)
    elif edge_count < target_edge_count:  
        lower_threshold = max(0, lower_threshold - 10)
        upper_threshold = max(50, upper_threshold - 10)
    return lower_threshold, upper_threshold


def capture_and_process():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Utilisation de DSHOW pour le backend
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Utilisation d'un seul processus pour la conversion en PPM
        convert_to_ppm(frame, 'Image/output_image.ppm')

        # Traiter l'image sans multiprocessing
        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_and_process()
