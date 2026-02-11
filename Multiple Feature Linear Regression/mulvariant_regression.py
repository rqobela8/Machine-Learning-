from data_actions import np_data
import numpy as np
import copy
import matplotlib.pyplot as plt

X_train,y_train = np_data()
print(X_train.shape)
w_init = np.random.uniform(-0.01, 0.01, size=X_train.shape[1])
b_init = 0.0


def train_data_infp():
    # data is stored in numpy array/matrix
    print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
    print(X_train)
    print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
    print(y_train)


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        cost = cost + (f_wb - y[i])**2

    cost = cost/(2*m)
    return cost




def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i,j]
        dj_db+=err

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw,dj_db



def gradient_descent(X,y,w_in,b_in,alpha,num_iters):
    """
       Performs batch gradient descent to learn w and b. Updates w and b by taking
       num_iters gradient steps with learning rate alpha

       Args:
         X (ndarray (m,n))   : Data, m examples with n features
         y (ndarray (m,))    : target values
         w_in (ndarray (n,)) : initial model parameters
         b_in (scalar)       : initial model parameter
         alpha (float)       : Learning rate
         num_iters (int)     : number of iterations to run gradient descent

       Returns:
         w (ndarray (n,)) : Updated values of parameters
         b (scalar)       : Updated value of parameter
         """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(X,y,w,b)

        w = w-alpha*dj_dw
        b = b - alpha*dj_db

        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(X, y, w, b))

    return w,b,J_history





if __name__ == '__main__':
    X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
    w_norm, b_norm, hist = gradient_descent(X_norm, y_train,w_init,b_init, 1.0e-2,1000 )
    m = X_norm.shape[0]
    yp = np.zeros(m)
    X_features = ['Hours_Worked', 'Sleep_Hours', 'Bugs', 'Deadline_Days', 'Coffee_Cups',
                  'Meetings', 'Interruptions', 'Remote_Work']
    dlc = {
        "dlorange": "#FF7F0E",
        "dlblue": "#1F77B4"
    }

    color = dlc["dlorange"]

    for i in range(m):
        yp[i] = np.dot(X_norm[i], w_norm) + b_norm

        # plot predictions and targets versus original features
    fig, ax = plt.subplots(1, 8, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train, label='target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:, i], yp, color=dlc["dlorange"], label='predict')
    ax[0].set_ylabel("Stress_Level")
    ax[0].legend()
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()



