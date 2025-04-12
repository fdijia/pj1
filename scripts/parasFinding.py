from matplotlib import pyplot as plt
import numpy as np
from classifyModel import NeuralNetwork, decode_model_name
from photoDataset import CIFAR10
from trainModel import train_model, save_history
import json
import os

train = CIFAR10('cifar-10-batches-py', train=True)
X_train, y_train = train.data / 255.0, np.eye(10)[train.labels]
test = CIFAR10('cifar-10-batches-py', train=False)
X_test, y_test = test.data / 255.0, np.eye(10)[test.labels]

def parasFinding():
    """不同层数参数组合"""
    paras1FindLr = [
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.02, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.03, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.04, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.05, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.06, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.08, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.09, 'reg_lambda': 0.001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.1, 'reg_lambda': 0.001},
    ]
    
    # return [paras1FindLr]
    
    paras1FindRl = [
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0001},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0003},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0007},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0009},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0011},
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0013}
    ]

    # return [paras1FindRl]

    paras1FindHidden = [
        {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
        {'layer_sizes': [3072, 256, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
        {'layer_sizes': [3072, 512, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
        {'layer_sizes': [3072, 1024, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
    ]
    
    # return [paras1FindHidden]
    paras2Layers = [
        {'layer_sizes': [3072, 512, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
        {'layer_sizes': [3072, 512, 64, 10], 'activations': ['relu', 'relu', 'softmax'], 'learning_rate': 0.07, 'reg_lambda': 0.0005},
        {'layer_sizes': [3072, 512, 128, 32, 10], 'activations': ['relu', 'relu', 'relu', 'softmax'], 'learning_rate': 0.007, 'reg_lambda': 0.0005}
    ]
    
    return [paras2Layers]


def compare_paras(paras_list, overwrite=False, modelSave=False):
    # 不保存模型，只保留记录，如果模型或记录存在，直接返回
    paraName = []
    
    for para in paras_list:
        model = NeuralNetwork(**para)
        paraName.append(model.name)
        isFile = os.path.exists('history/' + model.name + '.json')
        if isFile:
            print(f"Model {model.name} already exists. Skipping training.")
        else:
            loss, val_acc = train_model(model, X_train, y_train, epochs=30, batch_size=200)
            test_acc = model.evaluate_model(X_test, y_test)
            if modelSave:
                model.saveModel()
            history = {
                'train_loss': loss,
                'val_acc': val_acc,
                'test_acc': test_acc,
            }
            save_history(history, model.name + '.json', overwrite=overwrite)
    
    return paraName

def plot_hyperparameter_performance(history_name, history_file=None, save_filename=None):
    """
    history_name: 模型名称列表
    history_file: 如果有文件名则读取文件，无则读取history目录下该模型历史
    save_filename: 保存图片的文件名
    """
    colors = plt.cm.tab20.colors[:12]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, name in enumerate(history_name):
        config = decode_model_name(name)
        lr = config['learning_rate']
        reg = config['reg_lambda']
        layer_sizes = config['layer_sizes']
        label = f"LR={lr}, λ={reg}\nLayers={'->'.join(list(map(str, layer_sizes)))}"
        
        fileName = history_file if history_file else 'history/' + name + '.json'
        
        with open(fileName) as f:
            metrics = json.load(f)
            
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
    if save_filename:
        plt.savefig('visualization/' + save_filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    
def main():
    paras = parasFinding()
    for i, para in enumerate(paras):
        paraName = compare_paras(para, modelSave=True)
        with open('visualization/'+f'paras2{i+1}.json', 'w') as f:
            json.dump(paraName, f, indent=4)
        plot_hyperparameter_performance(paraName, save_filename=f'paras2{i+1}.png')
    

if __name__ == '__main__':
    main()
    # with open('visualization/paras1.json', 'r') as f:
    #     paraName = json.load(f)
    # plot_hyperparameter_performance(paraName, save_filename=f'paras1.png')