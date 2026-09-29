
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "mnist_logistic_model.pth"
OUTPUT_DIR = BASE_DIR / "outputs"

class MNISTLogisticRegression(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

def run_sample_prediction():
    # Load Trained Weights
    model = MNISTLogisticRegression()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # Get sample image from test dataset
    test_dataset = datasets.MNIST(
        root=BASE_DIR / "data", 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    image_tensor, true_label = test_dataset[0]  # Select 1st sample
    input_tensor = image_tensor.view(1, 784)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()

    print("\n--- Prediction Output ---")
    print(f"True Digit     : {true_label}")
    print(f"Predicted Digit: {prediction}")
    print(f"Confidence     : {confidence:.2%}")

    # Generate and Save Sample Output Plot
    plt.figure(figsize=(4, 4))
    plt.imshow(image_tensor.squeeze().numpy(), cmap="gray")
    plt.title(f"True: {true_label} | Pred: {prediction} ({confidence:.1%})")
    plt.axis("off")
    plt.tight_layout()
    sample_img_path = OUTPUT_DIR / "prediction_sample.png"
    plt.savefig(sample_img_path, dpi=150)
    plt.close()
    print(f"Visualization saved to: {sample_img_path}")

if __name__ == "__main__":
    run_sample_prediction()
