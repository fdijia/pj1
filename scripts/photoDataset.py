import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class CIFAR10():
    def __init__(self, root_dir, train=True):
        with open(os.path.join(root_dir, 'batches.meta'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            self.label_names = [name.decode('utf-8') for name in batch[b'label_names']]

        self.data_files = []
        if train:
            for i in range(1, 6):
                self.data_files.append(os.path.join(root_dir, f'data_batch_{i}'))
        else:
            self.data_files.append(os.path.join(root_dir, 'test_batch'))

        self.data = []
        self.labels = []

        for file in self.data_files:
                with open(file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.extend(batch[b'data'])
                    self.labels.extend(batch[b'labels'])
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        label = self.labels[idx] 
        return image_data, label

    def visualize(self, idx):
        image_data, label = self.__getitem__(idx)
        image = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
        label_name = self.label_names[label]
        plt.imshow(image)
        plt.axis('off')
        plt.title(label_name)
        plt.show()


if __name__ == "__main__":
    train_dataset = CIFAR10('cifar-10-batches-py', train=False)
    train_dataset.visualize(0)