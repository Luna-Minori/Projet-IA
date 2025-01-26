import cv2
from multiprocessing import Process, Queue
import numpy as np
import os

# Fonction pour capturer la vidéo
def capture_video(queue):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Utilisation de DSHOW pour le backend
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la caméra.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la frame.")
            break
        
        # Mettre l'image dans la queue pour la conversion
        queue.put(frame)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer le flou gaussien pour réduire le bruit
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Calculer les seuils dynamiques de base
        lower_threshold, upper_threshold = calculate_dynamic_thresholds(blurred_frame)

        # Appliquer la détection des contours avec les seuils dynamiques
        edges = cv2.Canny(blurred_frame, lower_threshold, upper_threshold)

        # Ajuster les seuils en fonction du nombre de bords détectés
        lower_threshold, upper_threshold = adjust_thresholds_based_on_edge_count(edges, lower_threshold, upper_threshold)

        # Appliquer à nouveau la détection avec les nouveaux seuils ajustés
        edges_adjusted = cv2.Canny(blurred_frame, lower_threshold, upper_threshold)

        # Afficher les résultats
        cv2.imshow('Original', frame)
        cv2.imshow('Gris', gray_frame)
        cv2.imshow('Flou Gaussien', blurred_frame)
        cv2.imshow('Contours Dynamiques Ajustés', edges_adjusted)
        
        print(f"Seuils dynamiques: {lower_threshold}, {upper_threshold}")  # Affiche les seuils pour le suivi
        print(f"Nombre de bords détectés: {np.sum(edges_adjusted > 0)}")

        # Quitter si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def calculate_dynamic_thresholds(image, lower_bound=50, upper_bound=200, target_edge_count=1000):
    """
    Calcule des seuils dynamiques pour la détection de contours Canny en fonction de l'image.
    L'algorithme ajuste les seuils en fonction du nombre de bords détectés.
    """
    # Calculer l'histogramme de l'image pour estimer l'intensité globale.
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Calculer la moyenne et l'écart type de l'image.
    mean_intensity = np.mean(image)
    std_dev_intensity = np.std(image)
    
    # Calculer des seuils de base en fonction de la moyenne et de l'écart type.
    lower_threshold = max(lower_bound, int(mean_intensity - 0.5 * std_dev_intensity))
    upper_threshold = min(upper_bound, int(mean_intensity + 0.5 * std_dev_intensity))
    
    # S'assurer que les seuils sont dans une plage valide
    lower_threshold = max(0, lower_threshold)
    upper_threshold = min(255, upper_threshold)
    
    return lower_threshold, upper_threshold

def adjust_thresholds_based_on_edge_count(edges, lower_threshold, upper_threshold, target_edge_count=10000):
    """
    Ajuste les seuils en fonction du nombre de bords détectés.
    Si trop de bords sont détectés, augmenter les seuils, sinon les réduire.
    """
    # Compter le nombre de pixels qui sont des bords (valeurs non nulles)
    edge_count = np.sum(edges > 0)
    
    # Ajuster les seuils en fonction du nombre de bords détectés
    if edge_count > target_edge_count:  # Trop de bords détectés
        lower_threshold = min(upper_threshold - 1, lower_threshold + 10)  # Augmenter le seuil inférieur
        upper_threshold = min(255, upper_threshold + 10)  # Augmenter le seuil supérieur
    elif edge_count < target_edge_count:  # Trop peu de bords détectés
        lower_threshold = max(0, lower_threshold - 10)  # Diminuer le seuil inférieur
        upper_threshold = max(50, upper_threshold - 10)  # Diminuer le seuil supérieur
    
    return lower_threshold, upper_threshold


# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    # Créer une queue pour la communication entre processus
    queue = Queue()

    # Démarrer le processus de capture vidéo
    video_process = Process(target=capture_video, args=(queue,))
    video_process.start()

    # Importer et démarrer le processus de conversion des images en PPM
    from image_convertion_ppm import convert_images
    convert_process = Process(target=convert_images, args=(queue,))
    convert_process.start()

    # Attendre la fin des processus
    video_process.join()
    convert_process.join()

