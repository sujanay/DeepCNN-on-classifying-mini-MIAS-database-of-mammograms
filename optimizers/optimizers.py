from keras import optimizers

def sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
    """
    Stochastic gradient descent optimizer
    :param lr: >=0, learning rate
    :param decay: >=0, learning rate decay over each update
    :param momentum: >=0, parameter that accelerates sgd in the relevant direction and dampens oscillations
    :param nesterov: boolean, whether to apply Nesterov momentum
    """
    return optimizers.sgd(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)

def rms_prop(lr=0.001, rho=0.9, epsilon=None, decay=0.0):
    """
    RMSProp optimizer
    :param lr: >=0, learning rate
    :param rho: >=0
    :param epsilon: >=0, fuzz factor. If None, defaults to K.epsilon()
    :param decay: >=0, learning rate decay over each update
    """
    return optimizers.rmsprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)

def adagrad(lr=0.01, epsilon=None, decay=0.0):
    """
    Adagrad optimizer
    :param lr: >=0, initial learning rate
    :param epsilon: >=0, If None, defaults to K.epsilon()
    :param decay: learning rate decay over each update
    """
    return optimizers.adagrad(lr=lr, epsilon=epsilon, decay=decay)

def adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0):
    """
    Adadelta optimizer
    :param lr: >=0, initial learning rate, defaults to 1. It is recommended to leave it at the default value
    :param rho: >=0, Adadelta decay factor, corresponding to fraction of gradient to keep at each time step
    :param epsilon: >=0, fuzz factor. If None, defaults to K.epsilon()
    :param decay: >=0, initial learning rate decay
    """
    return optimizers.adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay)

def adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False):
    """
    Adam optimizer
    :param lr: >=0, learning rate
    :param beta_1: 0 < beta_1 < 1, generally close to 1
    :param beta_2: 0 < beta_2 < 1, generally close to 1
    :param epsilon: >=0, fuzz factor. If None, defaults to K.epsilon()
    :param decay: >=0, learning rate decay over each update
    :param amsgrad: boolean, Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".
    """
    return optimizers.adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)

def adammax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0):
    """
    Adam optimizer
    :param lr: >=0, learning rate
    :param beta_1: 0 < beta_1 < 1, generally close to 1
    :param beta_2: 0 < beta_2 < 1, generally close to 1
    :param epsilon: >=0, fuzz factor. If None, defaults to K.epsilon()
    :param decay: >=0, learning rate decay over each update
    """
    return optimizers.adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)

def nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004):
    """
    Adam optimizer
    :param lr: >=0, learning rate
    :param beta_1: 0 < beta_1 < 1, generally close to 1
    :param beta_2: 0 < beta_2 < 1, generally close to 1
    :param epsilon: >=0, fuzz factor. If None, defaults to K.epsilon()
    """
    return optimizers.nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, schedule_decay=schedule_decay)

