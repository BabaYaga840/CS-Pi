"""Download the released assets from the Hugging Face Hub into ./assets:

    checkpoints/{avoid2d,spheres,orient,shelf}.pt   the four frozen flow priors used in the paper
    tasks/{spheres,orient,shelf}.pt                 evaluation tasks (start / goal configurations)
    data/avoid2d_support.npz                        demonstration-support raster for the 2D metric
    data/*_train.pt                                 training data (only needed for train.sh)

    python scripts/download_assets.py [--no-train-data]
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO = os.environ.get("CSPI_HF_REPO", "Soumojit048/cspi")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train-data", action="store_true")
    a = ap.parse_args()
    ignore = ["data/*_train.pt"] if a.no_train_data else None
    path = snapshot_download(repo_id=REPO, local_dir=os.environ.get("CSPI_ASSETS", "assets"), ignore_patterns=ignore)
    print("assets in", path)
