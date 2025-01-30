import numpy as np

class Neurone:
    def __init__(self, alpha):
        self.poids = []  # Poids du neurone
        self.seuil = 0  # Seuil (biais)
        self.entry = []  # Entrées du neurone
        self.sortie = 0  # Sortie du neurone
        self.alpha = alpha  # Paramètre pour Leaky ReLU
    
    def setneurone(self, poids, seuil, entry):
        self.poids = poids
        self.seuil = seuil
        self.entry = entry
    
    def calculs(self):
        # Calcul de la somme pondérée des entrées
        somme = np.dot(self.entry, self.poids) + self.seuil  # Somme pondérée avec seuil
        self.sortie = max(0, somme) if somme > 0 else self.alpha * somme  # Leaky ReLU

class Couche:
    def __init__(self, nbneurones, nbentry, alpha):
        self.neurones = []
        self.nbneurones = nbneurones
        self.alpha = alpha
    
    def setcouche(self, poids, seuil, entry):
        for i in range(self.nbneurones):
            neurone = Neurone(self.alpha)
            neurone.setneurone(poids[i], seuil[i], entry)
            self.neurones.append(neurone)
    
    def calculs(self):
        sorties = []
        for neurone in self.neurones:
            neurone.calculs()
            sorties.append(neurone.sortie)
        return sorties

class Reseau:
    def __init__(self, alpha):
        self.couches = []
        self.alpha = alpha
        
    def ajouter_couche(self, nbneurones, nbentry, poids, seuil):
        couche = Couche(nbneurones, nbentry, self.alpha)
        couche.setcouche(poids, seuil, [])
        self.couches.append(couche)
    
    def propagation_avant(self, image_entrée):
        entree = image_entrée
        for couche in self.couches:
            # Calcul de la sortie de chaque couche
            entree = couche.calculs()
        return entree

def load_ppm(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    largeur, hauteur = map(int, lines[1].split())
    expected_size = largeur * hauteur * 3
    data = list(map(int, " ".join(lines[3:]).split()))
    
    if len(data) != expected_size:
        raise ValueError(f"Taille inattendue: {len(data)} éléments au lieu de {expected_size}.")
    
    return data, largeur, hauteur

