from matplotlib import pyplot as plt
import numpy as np
import os

def training_plots(total_loss, accuracies, num_epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(np.arange(num_epochs), total_loss, 'g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    
    ax2.plot(np.arange(num_epochs), accuracies, 'b')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_plots.png')