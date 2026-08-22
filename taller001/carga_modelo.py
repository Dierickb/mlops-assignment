import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "decision_tree_penguins.pkl"

with open("decision_tree_penguins.pkl", "rb") as f:
    model = pickle.load(f)