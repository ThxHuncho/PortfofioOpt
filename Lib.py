import subprocess
import sys

# Liste des bibliothèques populaires à installer, y compris cvxpy et tqdm
libraries = [
    'numpy',       # Calcul scientifique
    'pandas',      # Manipulation et analyse de données
    'matplotlib',  # Visualisation de données
    'scikit-learn',# Machine learning
    'requests',    # HTTP requests
    'flask',       # Web framework
    'django',      # Web framework
    'beautifulsoup4',  # Web scraping
    'seaborn',     # Visualisation avancée
    'tensorflow',  # Machine learning / Deep learning
    'keras',       # High-level API for TensorFlow
    'scipy',       # Calcul scientifique
    'pytest',      # Tests unitaires
    'sqlalchemy',  # ORM pour bases de données
    'openpyxl',    # Manipulation de fichiers Excel
    'Pillow',      # Traitement d'image
    'cvxpy',       # Optimisation convexe
    'tqdm',        # Barre de progression pour boucles
    'yfinance'
]

def install_libraries():
    for lib in libraries:
        try:
            # Vérifie si la bibliothèque est déjà installée et l'installe si nécessaire
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"{lib} a été installé avec succès.")
        except subprocess.CalledProcessError:
            print(f"Erreur lors de l'installation de {lib}.")

if __name__ == "__main__":
    install_libraries()
