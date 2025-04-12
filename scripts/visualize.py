import seaborn as sns
import matplotlib.pyplot as plt
from classifyModel import NeuralNetwork
import parasFinding
import json

def plot_weight_heatmap(W, layer_name='W', save_path=None):
    plt.figure(figsize=(12, 8))
    sns.heatmap(W,
                cmap='coolwarm',
                center=0,
                xticklabels=False,
                yticklabels=False)
    plt.title(f'"{layer_name}" Weight Heatmap')
    if save_path:
        plt.savefig(f'./visualization/{save_path}.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()

# def visualize_history(json_filename, save_filename=None):
#     """只用来可视化model中的.npz.json文件的训练历史 即bestmodel的训练历史，图片保存在visualization文件夹中"""
#     parasFinding.visualize_history(json_filename, save_filename)
    
def visualize_compare_history(json_filename, save_filename=None):
    """
    可视化visualization中的json文件的训练历史，在一个图中进行比较
    json_filename：列表，包含其中各模型名字
    如果save_filename不为空，图片保存在visualization/save_filename文件夹中
    """
    if 'visualization/' not in json_filename:
        json_filename = 'visualization/' + json_filename
    with open(json_filename, 'r') as f:
        parasName = json.load(f)
    parasFinding.plot_hyperparameter_performance(parasName, save_filename)


if __name__ == '__main__':
    para = {'layer_sizes': [3072, 512, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.007, 'reg_lambda': 0.0005}
    model = NeuralNetwork(**para)
    plot_weight_heatmap(model.W[0], layer_name='W1', save_path='W1')
    plot_weight_heatmap(model.W[1], layer_name='W2', save_path='W2')
    
    visualize_compare_history('paras1FindLr1.json')
