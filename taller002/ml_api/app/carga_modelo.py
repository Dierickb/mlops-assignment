import pickle
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

def cargar_modelos():
    modelos = {}
    
    # Modelo 1: Arbol de Decision
    tree_path = MODEL_DIR / "decision_tree_penguins.pkl"
    if tree_path.exists():
        with open(tree_path, "rb") as f:
            modelos["decision_tree"] = pickle.load(f)
            
    # Modelo 2: SVM
    svm_path = MODEL_DIR / "svm_penguins.pkl"
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            modelos["svm"] = pickle.load(f)
            
    return modelos
