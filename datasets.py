
import os, gzip, cPickle
import numpy as np

# Onehot the targets
def onehot(x,numclasses=None):
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = np.max(x) + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype="int")
    z = np.zeros(x.shape, dtype="int")
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z
    result = np.reshape(result,(np.shape(result)[0], np.shape(result)[result.ndim-1]))
    return result

def load_mnist(*args, **kwargs):
    # Download the MNIST dataset if it is not present
    dataset = 'data/datasets/mnist_28x28.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            ".",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    x, y = np.meshgrid(range(28), range(28))
    c = (x + y) % 2 == 0
    mmean = train_set[0].mean(axis=0)
    mmask = np.logical_and(mmean>.027, c.flatten())
    #mmask = np.zeros((28, 28), dtype=np.bool)
    #mmask[::2, ::2] = True
    #mmask = mmask.flatten()
    train_set_x = train_set[0][:, mmask]
    valid_set_x = valid_set[0][:, mmask]
    test_set_x = test_set[0][:, mmask]
    train_set_y = np.float32(onehot(train_set[1]))
    valid_set_y = np.float32(onehot(valid_set[1]))
    test_set_y = np.float32(onehot(test_set[1]))
    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_tidigits(rndseed=42):
    ''' load and prepare TIDIGITS data '''
    cat = lambda y: np.hstack((y[:, [0]] + y[:, [1]], y[:, 2:])) # merge first two classes
    dat = np.load('data/datasets/tidigits.npz')
    x_train_key = 'arr_0'
    y_train_key = 'arr_3'
    x_test_key = 'arr_1'
    y_test_key = 'arr_4'
    ids = range(len(dat[x_train_key]))
    ids_test = range(len(dat[x_test_key]))
    np.random.seed(rndseed)
    np.random.shuffle(ids)
    np.random.shuffle(ids_test)
    np.savetxt('data/tmp/ids.dat', ids)
    np.savetxt('data/tmp/ids_test.dat', ids_test)
    train_x = - dat[x_train_key][ids[:-2000]] + 1
    train_y = dat[y_train_key][ids[:-2000]] + .5
    valid_x = - dat[x_train_key][ids[-2000:]] + 1
    valid_y = dat[y_train_key][ids[-2000:]] + .5
    test_x = - dat[x_test_key][ids_test] + 1
    test_y = dat[y_test_key][ids_test] + .5

    #whiten data
    sigma = np.dot(train_x.T, train_x)
    U, S, V = np.linalg.svd(sigma)
    principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
    #train_x = np.dot(train_x, principal_components)
    #valid_x = np.dot(valid_x, principal_components)
    #test_x = np.dot(test_x, principal_components)
    #train_x += 1.5 * train_x.std()
    #valid_x += 1.5 * train_x.std()
    #test_x +=  1.5 * train_x.std()

    #shift data
    train_min = train_x.min(axis=0)
    train_x -= train_min
    valid_x -= train_min
    test_x -= train_min
    train_max = train_x.max(axis=0)
    train_x /= train_max
    valid_x /= train_max
    test_x /= train_max

    return [(train_x, train_y),
            (valid_x, valid_y),
            (test_x, test_y)]
