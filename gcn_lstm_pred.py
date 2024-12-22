import numpy as np
import matplotlib.pyplot as plt
from config import *

# Load the arrays from files
testX = np.load(testX_npy)
predictions = np.load(predictions_npy)

print("testX and predictions have been loaded from files.")

# Plot predicted vs. actual values
plt.figure(figsize=(10, 5))
plt.plot(range(len(testX)), testX.flatten(), label='Actual Test Values', marker='o')
plt.plot(range(len(predictions)), predictions.flatten(), label='Predicted Test Values', marker='x')
plt.title('Actual vs Predicted Test Values')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()