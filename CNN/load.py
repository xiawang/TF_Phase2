from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np

def reformat(samples, labels):
    new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)
    labels = np.array([x[0] for x in labels])   # slow code, whatever
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels

def normalize(samples):
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a/3.0
    return a/128.0 - 1.0


def distribution(labels, name):
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()

def inspect(dataset, labels, i):
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset = dataset.reshape(shape[0], shape[1], shape[2])
    print(labels[i])
    plt.imshow(dataset[i])
    plt.show()


train = load('./data/train_32x32.mat')
test = load('./data/test_32x32.mat')
# extra = load('../data/extra_32x32.mat')

# print('Train Samples Shape:', train['X'].shape)
# print('Train  Labels Shape:', train['y'].shape)

# print('Train Samples Shape:', test['X'].shape)
# print('Train  Labels Shape:', test['y'].shape)

# print('Train Samples Shape:', extra['X'].shape)
# print('Train  Labels Shape:', extra['y'].shape)

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
# extra_samples = extra['X']
# extra_labels = extra['y']

n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)

num_labels = 10
image_size = 32
num_channels = 1
