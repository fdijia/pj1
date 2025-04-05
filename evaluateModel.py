from classifyModel import NeuralNetwork, decode_model_name
from photoDataset import CIFAR10
import numpy as np
import os
import json
test = CIFAR10('cifar-10-batches-py', train=False)
X_test, y_test = test.data / 255.0, np.eye(10)[test.labels]

def find_file():
    files = []
    for file in os.listdir('.'):
        if file.endswith('.json'):
            files.append(file)
    return files
        
def test_file(name, acc):
    config = decode_model_name(name)
    lr = config['learning_rate']
    reg = config['reg_lambda']
    layer_sizes = config['layer_sizes']
    label = f"LR={lr}, λ={reg}\nLayers={'->'.join(list(map(str, layer_sizes)))}"

    print("model of", label, ", accuracy:", acc)
    return acc


if __name__ == "__main__":
    for name in find_file():
        data = json.load(open(name))
        for key in data:
            test_file(key, data[key]['test_acc'])
