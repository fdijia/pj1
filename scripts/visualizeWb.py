import seaborn as sns
import matplotlib.pyplot as plt
from classifyModel import NeuralNetwork

def plot_weight_heatmap(W, layer_name):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cmap='coolwarm',
                center=0,
                xticklabels=False,
                yticklabels=False)
    plt.title(f'"{layer_name}" Weight Heatmap')
    plt.savefig(f'./visualization/{layer_name}_weights_heatmap.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    para = {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001}
    model = NeuralNetwork(**para)
    plot_weight_heatmap(model.W[0], 'W1')
    plot_weight_heatmap(model.W[1], 'W2')
