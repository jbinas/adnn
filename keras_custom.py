
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from round_op import GradPreserveRoundTensor
from round_weights import round_weights
import numpy as np


class ParameterizedLayer(Layer):
    ''' Parameterized fully connected relu layer
    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        #self.init = initializations.get(init)
        self.activation = self.param_relu
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        _i = 0
        while 'params_%s' % _i in kwargs:
            self.__setattr__('params_%s' % _i, kwargs.pop('params_%s' % _i))
            _i += 1

        self.ifactor = kwargs.pop('ifactor', 1.)
        self.init_scale = kwargs.pop('init_scale', (-.5, .5))
        self.integer_bits = kwargs.pop('integer_bits', 1)
        self.fractional_bits = kwargs.pop('fractional_bits', 2)

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ParameterizedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def init(self, shape, name=None):
        return K.variable(np.random.uniform(low=self.init_scale[0], high=self.init_scale[1], size=shape), name=name)

    def call(self, x, mask=None):
        output = K.dot(x, self.apply_precision(self.W))
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim,
                  }
        base_config = super(ParameterizedLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def param_relu(self, x):
        ''' Pass through parameterized activation '''
        return self.params_0 * K.relu(x)

    def apply_precision(self, X):
        fractional_shift = K.cast_to_floatx(2. ** (self.fractional_bits))
        max_limit = K.cast_to_floatx((2. ** (self.integer_bits + self.fractional_bits)) - 1)
        value = K.T.where(X < 0, X / self.ifactor, X * 1.)
        value = value * fractional_shift
        value = GradPreserveRoundTensor(value)
        value = K.clip(value, -max_limit, max_limit)
        value = value / fractional_shift
        value = K.T.where(value < 0, value * self.ifactor, value)
        return value


class PlasticReLU(Layer):
    '''Parametric Rectified Linear Unit:
    `f(x) = alphas * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alphas` is a learned array with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        init: initialization function for the weights.
        weights: initial weights, as a list of a single numpy array.
    '''
    def __init__(self, init='one', weights=None, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(PlasticReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alphas = self.init(input_shape[1:],
                                name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return self.alphas * K.relu(x)

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(PlasticReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ActivitySparseness(regularizers.Regularizer):
    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        output = self.layer.output
        #regularized_loss = loss + self.l1 * K.sum(K.mean(K.abs(output),axis=0))
        active = output - K.mean(output, axis=1, keepdims=True)
        active = (K.abs(active) + active) / 2.
        regularized_loss = loss + self.l1 * K.sum(K.mean(active, axis=0))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': self.l1}


#custom objectives
def act_sq_log_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean( (1 + 3*y_true) * K.square(first_log - second_log), axis=-1)

def act_sq_error(y_true, y_pred):
    return K.mean( (.25 + .75*y_true) * K.square(y_pred - y_true), axis=-1)


#constraints
class HardBounds(constraints.Constraint):
    '''Implements hard lower and upper bounds'''
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, p):
        p *= K.cast((p >= self.vmin) * (p <= self.vmax), K.floatx())
        return p


#regularizers
class Inh_l1(regularizers.Regularizer):
    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss + K.sum(K.abs(self.p) - self.p) / 2. * self.l1
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': self.l1}


