import os


def runge_kutta(f,y0:float,interval:tuple,n:int=10)->list:
    """
    Numerically solves an ordinary differential equation (ODE) using the classical
    4th-order Runge-Kutta method (RK4).

    Parameters
    ----------
    f : callable
        The derivative function of the ODE, with signature f(x, y).
        It must accept two arguments:
            x : float
                The independent variable.
            y : float
                The dependent variable (current estimate of y at x).

    y0 : float
        Initial value of y at the starting point of the interval (y(x0)).

    interval : tuple of float
        A tuple (x0, xn) representing the domain over which to solve the ODE:
            x0 : float
                Start of the interval.
            xn : float
                End of the interval.

    n : int, optional (default=10)
        Number of steps (subintervals) to divide the interval [x0, xn] into.
        A higher value increases accuracy but requires more computation.

    Returns
    -------
    list of tuple
        A list of (x, y) tuples, where each tuple represents the estimated solution y at a corresponding x.
        The list has (n) points starting from x0 to xn.

    Notes
    -----
    This implementation uses the classical Runge-Kutta 4th-order method:
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6

    Example
    -------
    >>> f = lambda x, y: 0.1 * y * (1 - y / 40)
    >>> runge_kutta(f, y0=1, interval=(1, 200), n=50)

    [(1.0, 1.0), (5.0, 1.452), ..., (200.0, 39.99)]
    """

    x=interval[0]
    h=(interval[1]-interval[0])/n
    y_est=y0
    curve=[]
    for i in range(1,n+1):
        k1=h*f(x=x,y=y_est)
        k2=h*f(x=x+h/2,y=y_est+k1/2)
        k3=h*f(x=x+h/2,y=y_est+k2/2)
        k4=h*f(x=x+h,y=y_est+k3)

        # Add point to curve
        curve.append( ( x, y_est) )

        # Update
        y_est+=(k1+2*k2+2*k3+k4)/6
        x+=h
    return curve

def build_dataset(t,y):
    t = t.reshape(-1, 1)
    y = y.reshape(-1, 1)

    split_idx = int(0.8 * len(t))
    t_train, y_train = t[:split_idx], y[:split_idx]
    t_test, y_test = t[split_idx:], y[split_idx:]

    dataset = dde.data.DataSet(
        X_train=t_train,
        y_train=y_train,
        X_test=t_test,
        y_test=y_test
    )

    return dataset

def pinn(a,b,y0,t0,tf,dataset):
    neurons = [50] + [50] + [50]
    layer_size = [1] + neurons + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    optimizer = "adam"
    learning_rate = 0.0001

    net = dde.nn.FNN(layer_size, activation, initializer)
    model = dde.Model(dataset, net)

    # resampler = dde.callbacks.PDEPointResampler(period=100)
    early_stop = dde.callbacks.EarlyStopping(
        monitor="loss_test",
        min_delta=1e-6,
        patience=5000,
    )

    model.compile(optimizer, lr=learning_rate, metrics=["l2 relative error"])
    os.makedirs('../../models', exist_ok=True)
    losshistory, train_state = model.train(iterations=20000, callbacks=[early_stop],
                                           model_save_path="../models/verhulst-2.1")

    dde.saveplot(losshistory, train_state, isplot=True, issave=False)

