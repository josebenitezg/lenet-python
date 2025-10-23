from matplotlib import pyplot as plt
import numpy as np
import os
from datasets import load_dataset

def training_plots(train_loss_list, train_accuracies, val_loss_list, val_accuracies, num_epochs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(np.arange(num_epochs), train_loss_list, 'g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    
    ax2.plot(np.arange(num_epochs), train_accuracies, 'b')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')

    ax3.plot(np.arange(num_epochs), val_loss_list, 'r')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Validation Loss over Epochs')

    ax4.plot(np.arange(num_epochs), val_accuracies, 'y')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy over Epochs')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_plots.png')

def download_data():
    print('Downloading data...')
    mnist = load_dataset("mnist")
    train_images = mnist['train']['image']
    train_labels = mnist['train']['label']
    test_images = mnist['test']['image']
    test_labels = mnist['test']['label']
    return train_images, train_labels, test_images, test_labels

def preprocess_data(images, labels, size):
    list_images = []
    list_labels = []
    for i in range(size):
        converted_image = np.array(images[i])
        list_images.append(converted_image)
        list_labels.append(labels[i])
    return list_images, list_labels