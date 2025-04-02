import numpy as np
import json
from classifyModel import NeuralNetwork
from photoDataset import CIFAR10
from sklearn.model_selection import train_test_split


def train_model(model, X_train, y_train, learning_rate=0.01, reg_lambda=0.01, epochs=100, batch_size=64):
    best_val_acc = 0
    best_params = None
    
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
            model.backward(X_batch, y_batch, learning_rate, reg_lambda)
        
        # 计算验证集准确率
        val_acc = evaluate_model(model, X_val, y_val)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'W1': model.W1,
                'b1': model.b1,
                'W2': model.W2,
                'b2': model.b2
            }
        
        # 学习率下降
        learning_rate *= 0.9

        print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.4f}')
    
    # 保存最佳模型参数
    txt_dict = {'W1': best_params['W1'].tolist(), 'b1': best_params['b1'].tolist(), 'W2': best_params['W2'].tolist(), 'b2': best_params['b2'].tolist()}
    with open('model.json', 'w') as f:
        json.dump(txt_dict, f, indent=4)

    model.W1, model.b1, model.W2, model.b2 = best_params['W1'], best_params['b1'], best_params['W2'], best_params['b2']
    return model


def evaluate_model(model, X, y):
    predictions = model.forward(X)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


if __name__ == "__main__":
    train = CIFAR10('cifar-10-batches-py', train=True)
    X_train, y_train = train.data / 255.0, np.eye(10)[train.labels]
    test = CIFAR10('cifar-10-batches-py', train=False)
    X_test, y_test = test.data / 255.0, np.eye(10)[test.labels]
    model = NeuralNetwork(input_size=32*32*3, hidden_size=60, output_size=10, activation='relu')
    train_model(model, X_train, y_train, learning_rate=0.1, epochs=100, batch_size=100)