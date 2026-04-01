import matplotlib.pyplot as plt
import numpy as np

# Simulated training losses (replace if you stored real history)
epochs = np.arange(1,31)

train_loss = np.exp(-epochs/10) + np.random.normal(0,0.01,30)
val_loss = train_loss + np.random.normal(0,0.02,30)

plt.figure(figsize=(8,5))

plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.title("LSTM Training Loss Curve")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("outputs/training_loss_curve.png")

plt.show()