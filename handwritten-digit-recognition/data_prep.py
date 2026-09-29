
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

# Convert to NumPy arrays & scale
X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
y_train = train_dataset.targets.numpy().astype(np.int64)

X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
y_test = test_dataset.targets.numpy().astype(np.int64)

# Flatten images to 784 features
X_train = X_train.reshape(len(X_train), 784)
X_test = X_test.reshape(len(X_test), 784)

# Data validation checks
if X_train.shape != (60000, 784) or X_test.shape != (10000, 784):
    raise ValueError("Incorrect dataset dimensions.")

if __name__ == "__main__":
    print("MNIST Dataset Summary")
    print("=" * 40)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"Pixel Range  : [{X_train.min()}, {X_train.max()}]")
