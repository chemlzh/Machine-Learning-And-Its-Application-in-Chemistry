import os
import struct
import numpy as np

# modified from https://cloud.tencent.com/developer/article/1194189
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def show_mnist_0to9(images, labels, num):
    """Show MNIST data images from 0 to 9, with the num_th group`"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )

    ax = ax.flatten()
    for i in range(10):
        img = images[labels == i][num].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig.suptitle('Num: %d' % num)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def show_mnist_OneImage(image, stitle):
    """Show MNIST data image, with text stitle`"""
    import matplotlib.pyplot as plt

    plt.imshow(image.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.title(stitle)
    plt.show()



# load dataset
#images, labels = load_mnist('./data/', 'train')
#images_test, labels_test = load_mnist('./data/', 't10k')

# show 0-9
#show_mnist_0to9(images, labels, 0)

# show one image
#i=5
#show_mnist_OneImage(images[i], 'No.%d, label=%d, prediction='%(i,labels[i]))

