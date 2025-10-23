from model import ConvNet
import numpy as np
from tqdm import tqdm
from utils import training_plots, download_data, preprocess_data

model = ConvNet()

LEARNING_RATE = 0.001
NUM_EPOCHS = 3
TRAIN_SIZE = 600 # TODO
TEST_SIZE = 100 # TODO

train_loss_list = []
train_accuracies = []
val_loss_list = []
val_accuracies = []

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = download_data()
    train_images, train_labels = preprocess_data(train_images, train_labels, TRAIN_SIZE)
    test_images, test_labels = preprocess_data(test_images, test_labels, TEST_SIZE)

    # FASE DE ENTRENAMIENTO
    for epoch in range(NUM_EPOCHS):
        train_total_loss = 0
        train_num_correct = 0
        val_total_loss = 0
        val_num_correct = 0
        print('Training...')
        for idx in tqdm(range(len(train_images))):
            x = train_images[idx]
            y = train_labels[idx]
            correct, loss = model.train_step(x, y, LEARNING_RATE)
            train_total_loss += loss
            if correct:
                train_num_correct += 1

        train_accuracy = train_num_correct / len(train_images)
        train_avg_loss = train_total_loss / len(train_images)
        

        train_loss_list.append(train_avg_loss)
        train_accuracies.append(train_accuracy)
        # FASE DE VALIDACIÓN
        # Solo evaluamos forward para no modificar los pesos
        print('Validation...')
        for idx in tqdm(range(len(test_images))):
            x = test_images[idx]
            y = test_labels[idx]

            y_pred = model.forward(x)
            loss = model.loss(y_pred, y)
            prediction = np.argmax(y_pred)
            val_total_loss += loss
            if prediction == y:
                val_num_correct += 1
        val_accuracy = val_num_correct / len(test_images)
        val_avg_loss = val_total_loss / len(test_images)
    
        val_loss_list.append(val_avg_loss)
        val_accuracies.append(val_accuracy)

        print(f'Train Accuracy: {train_accuracy}, Train Loss: {train_avg_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy}, Validation Loss: {val_avg_loss:.4f}')

    training_plots(train_loss_list, train_accuracies, val_loss_list, val_accuracies, NUM_EPOCHS)