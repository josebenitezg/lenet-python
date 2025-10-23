from model import ConvNet
from datasets import load_dataset
from tqdm import tqdm
from utils import training_plots
import numpy as np

mnist = load_dataset("mnist")

model = ConvNet()

num_samples = 100
learning_rate = 0.001
num_epochs = 10

total_loss_list = []
accuracies = []

for epoch in range(num_epochs):
    total_loss = 0
    num_correct = 0
    for idx in tqdm(range(num_samples)):
        x = mnist['train'][idx]['image']
        x = np.array(x)
        y = mnist['train'][idx]['label']
        correct, loss = model.train_step(x, y, learning_rate)
        total_loss += loss
        if correct:
            num_correct += 1

    accuracy = num_correct / num_samples
    avg_loss = total_loss / num_samples
    print(f'Epoch {epoch}, Accuracy: {accuracy}, Loss: {avg_loss}')

    total_loss_list.append(avg_loss)
    accuracies.append(accuracy)
    print(total_loss_list)

training_plots(total_loss_list, accuracies, num_epochs)
