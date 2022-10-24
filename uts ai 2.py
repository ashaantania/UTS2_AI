# Asha Antania (21091397068)

# insialisasi numpys
import numpy as np

# inisialisasi variable
# Input layer feature 10
# Per batchnya 6 input
inputs = [
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9, 5.0, 5.5],
    [1.5, 1.6, 2.0, 2.2, 2.4, 3.0, 3.6, 4.0, 4.5, 5.2],
    [9.2, 4.2, 1.3, 8.2, 2.4, 8.4, 5.8, 7.4, 1.6, 9.3],
    [2.8, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
    [2.5, 1.8, 2.0, 2.6, 2.8, 3.6, 4.0, 4.6, 5.0, 5.6],
    [2.0, 6.0, 7.0, 7.2, 8.0, 8.2, 9.0, 9.2, 10.6, 10.8],
]

# inisialisasi bobot variable
# jumlah weight sesuai dengan jumlah neuron layer1, yaitu 5
weights1 = [
    [2.5, 1.8, 2.0, 2.6, 2.8, 3.6, 4.0, 4.6, 5.0, 5.6],
    [1.5, 3.4, 0.9, 3.2, 0.4, 0.1, 2.8, 6.2, 8.4, 3.7],
    [1.0, 1.5, 2.4, 2.8, 3.0, 3.2, 4.0, 4.2, 5.4, 5.8],
    [2.5, 2.0, 2.2, 2.4, 3.0, 3.4, 4.0, 4.6, 5.2, 5.5],
    [2.0, 6.5, 7.8, 7.2, 8.4, 8.6, 9.0, 9.6, 10.0, 10.6],
]

# inisialisasi bias
# jumlah bias pada layer1, yaitu 5
bias1 = [1.5, 2.0, 3.0, 4.5, 5.5]

# panjang weights sesuai dengan neuron layer1, yaitu 5
# jumlah weights sesuai dengan jumlah neuron layer2, yaitu 3
weights2 = [
    [2.0, 7.6, 4.6, 1.5, 8.0],
    [3.1, 2.5, 1.8, 6.5, 7.6],
    [1.5, 7.8, 6.5, 2.8, 8.6]
]

# jumlah bias pada layer2, yaitu 3 neuron
bias2 = [3.5, 1.5, 4.0]

# command untuk menghitung layer1 menggunakan inputs, weights1, dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1) . T) + bias1

# command untuk menghitung layer2 menggunakan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2) . T) + bias2

# mencetak output layer2
print(layer2_outputs)