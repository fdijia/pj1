from matplotlib import pyplot as plt
import numpy as np
from classifyModel import NeuralNetwork, decode_model_name
from photoDataset import CIFAR10
from trainModel import train_model
import json

train = CIFAR10('cifar-10-batches-py', train=True)
X_train, y_train = train.data / 255.0, np.eye(10)[train.labels]
test = CIFAR10('cifar-10-batches-py', train=False)
X_test, y_test = test.data / 255.0, np.eye(10)[test.labels]

def parasFinding():
    """不同层数参数组合"""
    paras1 = [
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.02, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.001, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.005},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 64, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.01}
    ]

    paras2 = [
        {'layer_sizes': [3072, 256, 64, 10], 'activations': ['relu', 'relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 256, 64, 10], 'activations': ['relu', 'relu', 'softmax'], 'learning_rate': 0.001, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 256, 64, 10], 'activations': ['relu', 'relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 32, 10], 'activations': ['relu', 'relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.01}
    ]

    paras3 = [
        {'layer_sizes': [3072, 512, 128, 32, 10], 'activations': ['relu', 'relu', 'relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 512, 128, 32, 10], 'activations': ['relu', 'relu', 'relu', 'softmax'], 'learning_rate': 0.001, 'reg_lambda': 0.01},
        {'layer_sizes': [3072, 512, 128, 32, 10], 'activations': ['relu', 'relu', 'relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 800, 200, 50, 10], 'activations': ['relu', 'relu', 'relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.01},
    ]
    return paras1, paras2, paras3


def compare_paras(paras_list):
    history = {}
    
    for para in paras_list:
        model = NeuralNetwork(**para)
        loss, val_acc = train_model(model, X_train, y_train, epochs=200, batch_size=200)
        test_acc = model.evaluate_model(X_test, y_test)
        history[model.name] = {'train_loss': loss, 'val_acc': val_acc, 'test_acc': test_acc}
    
    return history

def plot_hyperparameter_performance(history, filename='para.png'):
    colors = plt.cm.tab20.colors[:12]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (name, metrics) in enumerate(history.items()):
        config = decode_model_name(name)
        lr = config['learning_rate']
        reg = config['reg_lambda']
        layer_sizes = config['layer_sizes']
        label = f"LR={lr}, λ={reg}\nLayers={'->'.join(list(map(str, layer_sizes)))}"
        epochs = np.arange(1, len(metrics['val_acc']) + 1)
    
        ax1.plot(epochs, metrics['val_acc'], 
                color=colors[i], 
                linewidth=2, 
                marker='o',
                label=label)
        
        ax2.plot(epochs, metrics['train_loss'],
                color=colors[i],
                linewidth=2,
                linestyle='--',
                marker='s',
                label=label)
    
    ax1.set_title('Validation Accuracy vs. Epochs', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_title('Training Loss vs. Epochs', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

 

def save_history_json(history, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"History saved to {filename}")
    except Exception as e:
        print(f"Error saving history: {str(e)}")


if __name__ == '__main__':
    paras = parasFinding()
    for i, para in enumerate(paras):
        history = compare_paras(para)
        save_history_json(history, f'./visualization/paras{i+1}.json')
        plot_hyperparameter_performance(history, filename=f'./visualization/paras{i+1}.png')