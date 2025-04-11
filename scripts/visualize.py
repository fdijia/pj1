import seaborn as sns
import matplotlib.pyplot as plt
from classifyModel import NeuralNetwork
import parasFinding
import json

def plot_weight_heatmap(W, layer_name='W'):
    plt.figure(figsize=(12, 8))
    sns.heatmap(W,
                cmap='coolwarm',
                center=0,
                xticklabels=False,
                yticklabels=False)
    plt.title(f'"{layer_name}" Weight Heatmap')
    plt.savefig(f'./visualization/{layer_name}_weights_heatmap.png', bbox_inches='tight', dpi=300)
    plt.show()

def visualize_history(json_filename, save_filename=None):
    """只用来可视化model中的.npz.json文件的训练历史 即bestmodel的训练历史，图片保存在visualization文件夹中"""
    parasFinding.visualize_history(json_filename, save_filename)
    
def visualize_compare_history(json_filename, save_filename=None):
    """可视化visualization中的json文件的训练历史，在一个图中进行比较，图片保存在visualization文件夹中"""
    if 'visualization/' not in json_filename:
        json_filename = 'visualization/' + json_filename
    with open(json_filename, 'r') as f:
        history = json.load(f)
    parasFinding.plot_hyperparameter_performance(history, save_filename)


if __name__ == '__main__':
    para = {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001}
    model = NeuralNetwork(**para)
    plot_weight_heatmap(model.W[0], 'W1')
    plot_weight_heatmap(model.W[1], 'W2')
    
    visualize_history('model3072r128_10lr0.01rl0.001.npz.json')
    
    visualize_compare_history('paras1.json')
