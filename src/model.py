import numpy as np

def ReLu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x)) # para evitar overflow
    return exp_x / np.sum(exp_x)

# Let's implement the full forward pass and the backward pass
class ConvNet:
    def __init__(self):
        # Convolutions - Kernels and biases
        self.weights_conv1 = np.random.randn(6, 5, 5) * 0.1
        self.bias_conv1 = np.zeros(6)
        self.weights_conv2 = np.random.randn(16, 5, 5, 6) * 0.1
        self.bias_conv2 = np.zeros(16) 
        # Fully connected - Weights and biases
        self.weights_fc1 = np.random.randn(120, 256) * 0.1
        self.bias_fc1 = np.zeros(120) 
        self.weights_fc2 = np.random.randn(84, 120) * 0.1
        self.bias_fc2 = np.zeros(84) 
        self.weights_fc3 = np.random.randn(10, 84) * 0.1
        self.bias_fc3 = np.zeros(10) 

    def forward(self, x):
        self.x = x
        # C1 --- CONVOLUTIONAL LAYER 1 ---
        '''
        input: 28x28x1
        output: 24x24x6
        '''
        self.conv1 = np.zeros((24, 24, 6))
        for k in range(6):
            for i in range(24):
                for j in range(24):
                    self.conv1[i, j, k] = np.sum(x[i:i+5, j:j+5] * self.weights_conv1[k]) + self.bias_conv1[k]
        # pool1 --- POOLING LAYER 1 ---
        '''
        input: 24x24x6
        output: 12x12x6
        '''
        self.pool1 = np.zeros((12, 12, 6))
        for k in range(6):
            for i in range(12):
                for j in range(12):
                    block = self.conv1[i*2:i*2+2, j*2:j*2+2, k]
                    self.pool1[i, j, k] = np.mean(block)
        
        # C2 --- CONVOLUTIONAL LAYER 2 ---
        '''
        input: 12x12x6
        output: 8x8x16
        '''
        self.conv2 = np.zeros((8, 8, 16))
        for k in range(16):
            for i in range(8):
                for j in range(8):
                    self.conv2[i, j, k] = np.sum(self.pool1[i:i+5, j:j+5, :] * self.weights_conv2[k]) + self.bias_conv2[k]
        # pool2
        self.pool2 = np.zeros((4, 4, 16))
        for k in range(16):
            for i in range(4):
                for j in range(4):
                    self.pool2[i, j, k] = np.mean(self.conv2[i*2:i*2+2, j*2:j*2+2, k])

        # flatten the results --- FLATTEN LAYER ---
        '''
        input: 4x4x16
        output: 256
        '''
        self.flatten = self.pool2.flatten()
        # FC z = w @ x + b - expresión matemática de la capa FC
        # FC1 --- FULLY CONNECTED LAYER 1 ---
        '''
        input: 256
        output: 120
        '''
        self.z1 = self.weights_fc1 @ self.flatten + self.bias_fc1
        self.output_fc1 = ReLu(self.z1)
        
        # FC2 --- FULLY CONNECTED LAYER 2 ---
        '''
        input: 120
        output: 84
        '''
        self.z2 = self.weights_fc2 @ self.output_fc1 + self.bias_fc2
        self.output_fc2 = ReLu(self.z2)
        
        # FC3 --- FULLY CONNECTED LAYER 3 ---
        '''
        input: 84
        output: 10 # numero de clases
        '''
        self.z3 = self.weights_fc3 @ self.output_fc2 + self.bias_fc3
        self.y_pred = softmax(self.z3)
        
        return self.y_pred

    def loss(self, output, label):
        return -np.log(output[label] + 1e-10)  # +epsilon para evitar log(0)
    
    def backward(self, label, learning_rate=0.01):
        # gradient of the loss with respect to the output
        # FC3: Sotfwmax + cross-entropy loss
        dz3 = self.y_pred.copy()
        dz3[label] -= 1 # para que el gradiente apunte hacia el valor correcto

        dWeights_fc3 = np.outer(dz3, self.output_fc2)
        dBiases_fc3 = dz3
        dOutput_fc2 = self.weights_fc3.T @ dz3

        # FC2: ReLu
        dz2 = dOutput_fc2 * (self.output_fc2 > 0)
        dWeights_fc2 = np.outer(dz2, self.output_fc1)
        dBiases_fc2 = dz2
        dOutput_fc1 = self.weights_fc2.T @ dz2

        # FC1: ReLu
        dz1 = dOutput_fc1 * (self.output_fc1 > 0)
        dWeights_fc1 = np.outer(dz1, self.flatten)
        dBiases_fc1 = dz1
        dFlatten = self.weights_fc1.T @ dz1

        # backprop for the conv layers
        # Paso 1: Reshape flatten → pool2
        dpool2 = dFlatten.reshape(4, 4, 16)

        # Paso 2: Backprop a través de Pool2
        dconv2 = np.zeros_like(self.conv2)
        for k in range(16):
            for i in range(4):
                for j in range(4):
                        # El gradiente se distribuye uniformemente
                        # en la región 2x2 que se promedió
                        dconv2[i*2:i*2+2, j*2:j*2+2, k] += dpool2[i, j, k] / 4
                        # Dividir entre 4 porque average pooling toma promedio de 4 valores
                    
        # Paso 3: Backprop a través de Conv2

        dweights_conv2 = np.zeros_like(self.weights_conv2)  # (16,5,5,6)
        dbias_conv2 = np.zeros_like(self.bias_conv2)        # (16,)
        dpool1 = np.zeros_like(self.pool1)                  # (12,12,6)
        
        for k in range(16):  # Para cada filtro
            for i in range(8):
                for j in range(8):
                    # Región del input que se usó en forward
                    input_patch = self.pool1[i:i+5, j:j+5, :]  # (5,5,6)
                    
                    # Gradiente del kernel: acumular sobre todas las posiciones
                    dweights_conv2[k] += input_patch * dconv2[i, j, k]
                    
                    # Gradiente del bias: simplemente sumar
                    dbias_conv2[k] += dconv2[i, j, k]
                    
                    # Gradiente hacia atrás (a pool1)
                    dpool1[i:i+5, j:j+5, :] += self.weights_conv2[k] * dconv2[i, j, k]
        # normalizar por numero de posiciones
        dweights_conv2 /= (8 * 8)
        dbias_conv2 /= (8 * 8)
        # Paso 4: Backprop a través de Pool1

        # ═══════════════════════════════════════════════════
        # POOL1: Average Pooling 2x2
        # Input: conv1 (24x24x6), Output: pool1 (12x12x6)
        # ═══════════════════════════════════════════════════
        dconv1 = np.zeros_like(self.conv1)  # (24,24,6)
        
        for k in range(6):
            for i in range(12):
                for j in range(12):
                    dconv1[i*2:i*2+2, j*2:j*2+2, k] += dpool1[i, j, k] / 4

        # Paso 5: Backprop a través de Conv1

        # ═══════════════════════════════════════════════════
        # CONV1: 28x28 → 24x24x6 (kernel 5x5, 6 filtros)
        # ═══════════════════════════════════════════════════
        dweights_conv1 = np.zeros_like(self.weights_conv1)  # (6,5,5)
        dbias_conv1 = np.zeros_like(self.bias_conv1)        # (6,)
        # d_input no lo necesitamos (no hay capas antes)
        
        for k in range(6):  # Para cada filtro
            for i in range(24):
                for j in range(24):
                    # Región del input que se usó
                    input_patch = self.x[i:i+5, j:j+5]  # (5,5)
                    
                    # Gradiente del kernel
                    dweights_conv1[k] += input_patch * dconv1[i, j, k]
                    
                    # Gradiente del bias
                    dbias_conv1[k] += dconv1[i, j, k]
        # normalizar por numero de posiciones
        dweights_conv1 /= (24 * 24)
        dbias_conv1 /= (24 * 24)
        
        # Actualizar pesos y sesgos de las capas FC
        self.weights_fc1 -= learning_rate * dWeights_fc1
        self.bias_fc1 -= learning_rate * dBiases_fc1
        self.weights_fc2 -= learning_rate * dWeights_fc2
        self.bias_fc2 -= learning_rate * dBiases_fc2
        self.weights_fc3 -= learning_rate * dWeights_fc3
        self.bias_fc3 -= learning_rate * dBiases_fc3

        # Actualizar pesos y sesgos de las capas convolucionales
        self.weights_conv2 -= learning_rate * dweights_conv2
        self.bias_conv2 -= learning_rate * dbias_conv2
        self.weights_conv1 -= learning_rate * dweights_conv1
        self.bias_conv1 -= learning_rate * dbias_conv1

    def train_step(self, x, y, learning_rate):
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.backward(y, learning_rate)
        prediction = np.argmax(y_pred)
        correct = prediction == y

        return correct, loss

