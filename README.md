# 从零开始构建三层神经网络分类器，实现图像分类
## 架构：
- **cifar-10-batches-py**          -- 储存数据集
- **models**                       -- 在parasFinding.py以及trainModel.py中训练的模型储存在这，可以通过文件名读取各参数，使用classifyModel.py中定义的函数解码
- **history**                      -- 存放训练历史，在训练后使用`save_history()`函数保存
- **scripts**                      -- 代码区域
  - **classfyModel.py**            -- 模型定义部分
  - **evaluateModel.py**           -- 模型评估部分，从已经训练完成后保存的文件中读取测试集上的准确率
  - **parasFinding.py**            -- 自定义各参数并训练和对比
  - **photoDatasets.py**           -- 数据区域，从cifar-10-batches-py中读取数据
  - **trainModel.py**              -- 训练模型部分
  - **visualize.py**               -- 可视化参数、训练历史
- **visualization**                -- 储存parasFingding.py中训练历史信息，每一组在一个json文件中存放了其中模型名字

## 模型训练
- `def train_model(model, X_train, y_train, epochs=100, batch_size=64) -> losses_history, val_acc_history`
  
  先将传入参数9：1分为训练集和验证集，在打乱数据每次训练batch_size个数据量，一共训练epochs次，每次训练都计算训练集上的损失和验证集上的准确率，并取准确率最好的一次训练作为最终模型，训练结束后不会自动保存，需要手动调用model.saveModel()保存。返回损失记录和准确率记录

## 文件使用说明
- 模型训练在`parasFinding.py`中进行，此时会以组为单位保存训练时的各模型名称，放在visualization文件夹下，同时里面还有数据可视化，训练损失以及测试集上的准确率则会保存在history中

- 要获取各个模型的测试集上的准确率只需运行`evaluateModel.py`，能自动读取models文件夹中所有模型并输出准确率以及各模型文件计算测试机准确率，可自行调整；或者可以运行其中的`findCompareHistory()`函数，可以对比`visualization`中json文件所有模型测试集上的的准确率（只是浏览历史记录而不是调用模型）。**不要再models文件夹中创建其他文件夹，只能读取models/与models/bestModel/中的.npz文件**

- 最终模型learning_rate=0.007，reg_lambda=0.0005，隐藏层为512，训练100epochs，在`trainModel.py`中训练得到最终模型。

- 可以通过运行`visualize.py`进行可视化，可以选择`visualization`文件夹下的json文件进行可视化，得到对比图；也能通过获取模型的W或b将其可视化

## 使用
- 将代码下载后保留`models`和`visualization`文件夹，但是其中内容可以删去
- 在`models`中有模型的情况下可以运行`evaluateModel.py`，可以查看目前存在模型在测试集上的准确率
- 在`visualization`中有`.json`格式文件情况下可以运行`visualize.py`中的`visualize_compare_history`函数，传入`.json`文件名称，可视化`parasFinding.py`中保存的模型对比历史
- `visualize.py`中的`plot_weight_heatmap`函数，传入模型的W或b可视化
- 将代码下载后可自行调整`parasFinding()`中的参数并运行`main`函数，就能对相应参数模型进行训练并保存训练历史于`history`文件夹中，该组训练模型名字保持在`visualization`文件夹中，可通过模型名字引用`history`中的训练历史。**注意要从头开始训练的话请将`models`中的模型删除（提供了model.delete()函数）**
- 如果找到了最佳参数，可通过调整`trainModel.train_best_model(para)`传入相关参数进行训练，训练后的模型以及训练历史保存在`models\bestModel`文件夹下，**注意如果原模型未删除会自动删除原模型**
- **注意使用模型时默认是`models`文件夹下的模型，如果想使用`models\bestModel`中的模型请在初始化时使用`model_name=bestModel/YOUR_MODEL_NAME`**
