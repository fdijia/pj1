import numpy as np
import json

def write_model():
    input_size = 500
    hidden_size = 100
    output_size = 10

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    txt_dict = {'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist()}
    with open('model.json', 'w') as f:
        json.dump(txt_dict, f, indent=4)

def read_model():
    with open('model.json', 'r') as f:
        txt_dict = json.load(f)
        w1 = np.array(txt_dict['b2'])
    return w1

if __name__ == "__main__":
    model = read_model()
    print(model)
    print(model.shape)
    print(type(model))