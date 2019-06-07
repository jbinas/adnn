
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras_custom import PlasticReLU, ParameterizedLayer
from datasets import load_mnist
from visualizer import Visualizer
import sys



if __name__ == '__main__':

    np.random.seed(int(sys.argv[1]))
    epochs = int(sys.argv[2])

    datasets = load_mnist(42)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    model = Sequential()
    #visualizer = Visualizer([196,196,10], model, 1.)

    model.add(ParameterizedLayer(
                input_dim=196, output_dim=100, init_scale=(-.5,.5),
                integer_bits=1, fractional_bits=2, ifactor=1.,
                params_0=np.ones(196, dtype='float32'),
                ))
    #model.add(PlasticReLU())
    #model.add(Dropout(0.5))
    model.add(ParameterizedLayer(
                input_dim=100, output_dim=50, init_scale=(-.5,.5),
                integer_bits=1, fractional_bits=2, ifactor=1.,
                params_0=np.ones(196, dtype='float32'),
                ))
    #model.add(PlasticReLU())
    #model.add(Dropout(0.5))
    model.add(Dense(10, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

    model.fit(train_set_x, train_set_y,
            validation_data=(valid_set_x, valid_set_y),
            nb_epoch=epochs,
            batch_size=200,
            #callbacks=[visualizer]
            )
    score = model.evaluate(test_set_x, test_set_y, batch_size=100)
    with open('out.dat', 'a') as f:
        f.write('%s\n' % score)

    #raw_input("Press enter to continue.")

