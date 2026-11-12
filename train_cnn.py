"""Train the Fashion-MNIST catalog CNN and save it to models/fashion_cnn.pt.

Usage:  python train_cnn.py [--epochs 4]
The dashboard loads these weights; if the file is missing it trains on first run.
"""
import argparse
import os

import torch

from aml_core import MODEL_PATH, dataset_to_tensors, load_fashion_mnist, train_cnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = train_cnn(epochs=args.epochs, seed=args.seed)
    x, y = dataset_to_tensors(load_fashion_mnist(train=False))
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"test accuracy: {acc * 100:.2f}%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved weights to {MODEL_PATH}")


if __name__ == "__main__":
    main()
