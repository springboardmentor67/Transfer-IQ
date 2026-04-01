import numpy as np
import matplotlib.pyplot as plt

# example error values
errors = np.random.normal(0,1,1000)

plt.figure(figsize=(8,5))

plt.hist(errors, bins=40)

plt.title("Prediction Error Distribution")

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.savefig("outputs/error_distribution.png")

plt.show()