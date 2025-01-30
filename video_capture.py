import cv2
from multiprocessing import Process, Queue
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Fonction pour capturer la vidéo et envoyer les images à la queue
def capture_video(queue):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la caméra.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la frame.")
            break

        # Envoyer l'image dans la queue pour le processus de conversion
        queue.put(frame)

        # Conversion en espace de couleur HSV
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv_image)

        # Conversion en niveaux de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer un flou gaussien pour réduire le bruit
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Calculer les seuils dynamiques de base
        lower_threshold, upper_threshold = calculate_dynamic_thresholds(blurred_frame)

        # Appliquer la détection des contours avec les seuils dynamiques
        edges = cv2.Canny(blurred_frame, lower_threshold, upper_threshold)

        # Ajuster les seuils dynamiquement
        lower_threshold, upper_threshold = adjust_thresholds_based_on_edge_count(edges, lower_threshold, upper_threshold)

        # Appliquer la détection avec les nouveaux seuils ajustés
        edges_adjusted = cv2.Canny(blurred_frame, lower_threshold, upper_threshold)

        # Définition des textes
        t_frame = f"Taille: {frame.shape[1]}x{frame.shape[0]}x{frame.shape[2]}, {frame.dtype}"
        t_hsv = f"Taille: {hsv_image.shape[1]}x{hsv_image.shape[0]}x{frame.shape[2]}, {hsv_image.dtype}"
        print("hue : ", hue , "\n\n", "saturation : ", saturation,"\n\n","value : ",value, "\n\n");
        # Position des textes
        p_frame = (70, frame.shape[0] - 20)
        p_hsv = (70, hsv_image.shape[0] - 20)

        # Couleur du texte (blanc)
        couleur = (255, 255, 255)

        # Ajouter un rectangle noir pour le bandeau
        cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(hsv_image, (0, hsv_image.shape[0] - 40), (hsv_image.shape[1], hsv_image.shape[0]), (0, 0, 0), -1)

        # Ajouter le texte sur les images
        cv2.putText(frame, t_frame, p_frame, cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2, cv2.LINE_AA)
        cv2.putText(hsv_image, t_hsv, p_hsv, cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2, cv2.LINE_AA)

        for i in range(50):
            frame[i,50] = 255,0,0
            frame[i,49] = 255,0,0
            frame[i,51] = 255,0,0

            frame[i,75] = 255,0,0
            frame[i,76] = 255,0,0
            frame[i,77] = 255,0,0

            frame[0,i+75] = 255,0,0
            frame[1,i+76] = 255,0,0
            frame[2,i+77] = 255,0,0

            if( i<30):
                frame[24,i+75] = 255,0,0
                frame[25,i+76] = 255,0,0
                frame[26,i+77] = 255,0,0

        cv2.circle(frame, (100, 100), 52, (0, 0, 0), -1)
        cv2.circle(frame, (100, 100), 50, (255, 255, 255), -1)
        frame[98:102, 98:102] = 0, 0, 0
        # Centre du cadran
        center = (100, 100)

        # Rayon du cadran
        radius = 50

        # Boucle pour dessiner les traits des heures
        for i in range(12):  # 12 heures
            # Calcul de l'angle pour chaque heure
            angle_deg = 360 / 12 * i  # Chaque heure est espacée de 30 degrés
            angle_rad = np.deg2rad(angle_deg)  # Conversion en radians

            # Calcul de la position du trait sur le cercle (en utilisant cos et sin)
            start_point = (int(center[0] + radius * np.cos(angle_rad)), 
                        int(center[1] + radius * np.sin(angle_rad)))
            
            # Longueur du trait
            line_length = 10

            end_point = (int(center[0] + (radius - line_length) * np.cos(angle_rad)), 
                        int(center[1] + (radius - line_length) * np.sin(angle_rad)))
            
            # Dessiner la ligne
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        now = datetime.datetime.now() #aiguille heure
        print("Heure actuelle: ", now.hour, ":", now.minute, ":", now.second)
        aiguille_angle_hour = np.radians((now.hour % 12) * 30 + now.minute * 0.5 - 90)
        x = int(100 + 15 * np.cos(aiguille_angle_hour)) 
        y = int(100 + 15 * np.sin(aiguille_angle_hour))
        cv2.line(frame, (100,100), (x,y), (0, 0, 0), 2)

        #aiguille minute
        aiguille_angle_min = np.radians(now.minute * 6 - 90)
        x_m = int(100 + 25 * np.cos(aiguille_angle_min)) 
        y_m = int(100 + 25 * np.sin(aiguille_angle_min))
        cv2.line(frame, (100,100), (x_m, y_m), (0, 0, 0), 2)
        
        cv2.imshow('Original', frame)
        cv2.imshow('HSV', hsv_image)
        cv2.imshow('Contours Ajustés', edges_adjusted)

        print(f"Seuils dynamiques: {lower_threshold}, {upper_threshold}")
        print(f"Nombre de bords détectés: {np.sum(edges_adjusted > 0)}")

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Fonction pour calculer les seuils dynamiques pour Canny
def calculate_dynamic_thresholds(image):
    mean_intensity = np.mean(image)
    std_dev_intensity = np.std(image)

    lower_threshold = max(50, int(mean_intensity - 0.5 * std_dev_intensity))
    upper_threshold = min(200, int(mean_intensity + 0.5 * std_dev_intensity))

    return lower_threshold, upper_threshold

# Fonction pour ajuster dynamiquement les seuils
def adjust_thresholds_based_on_edge_count(edges, lower_threshold, upper_threshold, target_edge_count=10000):
    edge_count = np.sum(edges > 0)

    if edge_count > target_edge_count:  # Trop de bords détectés
        lower_threshold = min(upper_threshold - 1, lower_threshold + 10)
        upper_threshold = min(255, upper_threshold + 10)
    elif edge_count < target_edge_count:  # Pas assez de bords détectés
        lower_threshold = max(0, lower_threshold - 10)
        upper_threshold = max(50, upper_threshold - 10)

    return lower_threshold, upper_threshold

# Fonction pour convertir les images en PPM (simulation)
def convert_images(queue):
    while True:
        if not queue.empty():
            frame = queue.get()

            # Conversion en PPM
            filename = f"frame_{os.getpid()}.ppm"
            cv2.imwrite(filename, frame)

            print(f"Image enregistrée : {filename}")

# Main
if __name__ == "__main__":
    queue = Queue()

    # Lancer la capture vidéo
    video_process = Process(target=capture_video, args=(queue,))
    video_process.start()

    # Lancer le processus de conversion en PPM
    convert_process = Process(target=convert_images, args=(queue,))
    convert_process.start()

    # Attendre la fin des processus
    video_process.join()
    convert_process.terminate()  # Arrête proprement le processus de conversion
