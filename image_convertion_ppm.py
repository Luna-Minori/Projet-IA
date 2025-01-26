import os
import time
from multiprocessing import Queue

# Fonction pour convertir une image en PPM
def convert_to_ppm(frame, output_image_path):
    height, width, channels = frame.shape
    with open(output_image_path, 'w') as file:
        file.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            for x in range(width):
                r, g, b = frame[y, x]
                file.write(f"{r} {g} {b} ")
            file.write("\n")
    print(f"L'image a été convertie en PPM et sauvegardée sous {output_image_path}")

# Fonction pour gérer la conversion des images à partir de la queue
def convert_images(queue):
    # Créer le répertoire pour sauvegarder les images PPM si ce n'est pas déjà fait
    output_dir = 'Image'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        if not queue.empty():
            frame = queue.get()
            # Générer un nom unique pour chaque image (basé sur l'heure actuelle)
            timestamp = int(time.time())
            output_image_path = os.path.join(output_dir, f'output_image_{timestamp}.ppm')
            convert_to_ppm(frame, output_image_path)
        time.sleep(10)  # Ajouter un léger délai pour réduire la charge CPU

