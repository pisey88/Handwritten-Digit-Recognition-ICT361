
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from data_prep import X_train, X_test, y_train, y_test, CLASS_NAMES, BASE_DIR

torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class MNISTLogisticRegression(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

model = MNISTLogisticRegression().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

train_losses = []
print(f"Using device: {DEVICE}\nTraining started...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    avg_loss = running_loss / len(train_dataset)
    train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] Loss: {avg_loss:.4f}")

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MNIST Multiclass Logistic Regression Loss")
plt.grid(True)
plt.tight_layout()
loss_path = OUTPUT_DIR / "training_loss.png"
plt.savefig(loss_path, dpi=150)
plt.close()

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        predictions = logits.argmax(dim=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predictions.cpu().numpy())

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")

report = classification_report(y_true, y_pred, labels=list(range(10)), target_names=CLASS_NAMES, zero_division=0)

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()

# Baseline Calculation
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

# Save Text Evaluation Summary
evaluation_path = OUTPUT_DIR / "evaluation_results.txt"
with open(evaluation_path, "w", encoding="utf-8") as f:
    f.write("MNIST Handwritten Digit Recognition Evaluation\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test Accuracy: {accuracy:.2%}\n")
    f.write(f"Baseline Accuracy: {baseline_accuracy:.2%}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save Model Weights
model_path = OUTPUT_DIR / "mnist_logistic_model.pth"
torch.save(model.state_dict(), model_path)

print(f"\nTraining Complete!")
print(f"Model saved to: {model_path}")
print(f"Results logged to: {OUTPUT_DIR}")
