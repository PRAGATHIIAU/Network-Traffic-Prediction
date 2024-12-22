import os

from matplotlib import pyplot as plt
import pandas as pd
from config import *


if os.path.exists(gcn_lstm_csv):
                print("hi")
                log_data = pd.read_csv(gcn_lstm_csv)
                # Plot the loss graph from CSVLogger data
                plt.plot(log_data['epoch'], log_data['loss'], label='Training Loss')
                plt.plot(log_data['epoch'], log_data['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss Over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                log_data = pd.read_csv(gcn_lstm_knn_csv)
                # Plot the loss graph from CSVLogger data
                plt.plot(log_data['epoch'], log_data['loss'], label='Training Loss')
                plt.plot(log_data['epoch'], log_data['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss Over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()