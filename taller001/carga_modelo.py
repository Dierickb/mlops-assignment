import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "modelos/decision_tree_penguins.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)