import numpy as np
from classifyModel import NeuralNetwork
from photoDataset import CIFAR10
from sklearn.model_selection import train_test_split
import json
import os

def train_model(model, X_train, y_train, epochs=100, batch_size=64):
    best_val_acc = 0
    best_params = None
    learning_rate = model.learning_rate
    losses_history = []
    val_acc_history = []

    for epoch in range(epochs):
        X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=49)

        # 打乱数据
        indices = np.arange(X_train_new.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train_new[indices]
        y_train_shuffled = y_train_new[indices]
        
        # 小批量梯度下降
        for i in range(0, X_train_new.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # 前向传播
            model.forward(X_batch)
            
            # 反向传播
            model.backward(X_batch, y_batch, learning_rate)
        
        # 计算训练集损失
        y_train_pred = model.forward(X_train)
        loss = model.compute_loss(y_train_pred, y_train)
        losses_history.append(loss)

        # 根据验证集指标自动保存最优的模型权重
        val_acc = model.evaluate_model(X_val, y_val)
        val_acc_history.append(val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_params = {'W': model.W,'b': model.b}
        
        # 学习率下降
        learning_rate *= 1 - np.sqrt(learning_rate / epochs)

        print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.4f}')
    
    # 保存最佳模型参数
    model.W, model.b = best_params['W'], best_params['b']
    save_dict = {}
    for i, (w, b) in enumerate(zip(best_params['W'], best_params['b'])):
        save_dict[f'W{i+1}'] = w
        save_dict[f'b{i+1}'] = b

    np.savez_compressed('./models/' + model.name, **save_dict)

    return losses_history, val_acc_history

def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {filename}")

def train_best_model():
    para = {'layer_sizes': [3072, 128, 10], 'activations': ['relu', 'softmax'], 'learning_rate': 0.01, 'reg_lambda': 0.001}
    model = NeuralNetwork(**para)
    learning_rate = [0.008, 0.003, 0.0008, 0.0003]
    train_loss = []
    val_acc = []
    for lr in learning_rate:
        model.changeParas(learning_rate=lr)
        loss, acc = train_model(model, X_train, y_train, epochs=100, batch_size=200)
        train_loss.extend(loss)
        val_acc.extend(acc)
    test_acc = model.evaluate_model(X_test, y_test)
    history = {'train_loss': train_loss, 'val_acc': val_acc, 'test_acc': test_acc}
    save_history(history, './models/' + model.name + '.json')


if __name__ == "__main__":
    # final train
    train = CIFAR10('cifar-10-batches-py', train=True)
    X_train, y_train = train.data / 255.0, np.eye(10)[train.labels]
    test = CIFAR10('cifar-10-batches-py', train=False)
    X_test, y_test = test.data / 255.0, np.eye(10)[test.labels]
    train_best_model()
   