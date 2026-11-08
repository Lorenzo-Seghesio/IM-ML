"""Model A: pretrained M1 PP/sensor embeddings with an XGBoost weight head."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utility.hybrid_pipeline import hybrid_main

if __name__ == "__main__":
    hybrid_main("gbdt_pppt")
