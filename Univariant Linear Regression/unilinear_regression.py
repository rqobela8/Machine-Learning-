
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os

def download_dataset():

    # Download latest version
    dir_path = kagglehub.dataset_download("poushal02/student-academic-stress-real-world-dataset")
    print("Path to dataset files:", dir_path)
    file_path = os.listdir(dir_path)[0]


    return os.path.join(dir_path,file_path)



def read_data():
    data_path = download_dataset()
    df = pd.read_csv(data_path)
    df=df[["Peer pressure","Academic pressure from your home",'Rate your academic stress index ']]

    #features
    peer_pressure = np.array(df["Peer pressure"].values)
    academic_pressure = np.array(df["Academic pressure from your home"].values)

    #Target
    stress_index = np.array(df['Rate your academic stress index '].values)
    print(academic_pressure,stress_index)

    return peer_pressure,academic_pressure,stress_index

def example_data():
    x_train = np.array([1.0, 2.0])  # features
    y_train = np.array([300.0, 500.0])  # target value
    return x_train,y_train

def compute_cost(w,b,feature,target):
    m = feature.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * feature[i] + b
        cost = cost + (f_wb - target[i])**2
    total_cost = 1/(2*m)*cost

    return total_cost

def compute_gradient(x,y,w,b):
    """
       Computes the gradient for linear regression
       Args:
         x (ndarray (m,)): Data, m examples
         y (ndarray (m,)): target values
         w,b (scalar)    : model parameters
       Returns
         dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
         dj_db (scalar): The gradient of the cost w.r.t. the parameter b
        """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    b = b_in
    w = w_in
    gradient_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        cost = compute_cost(w,b,x,y)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        gradient_history.append({"w":w,"b":b,"cost":cost})

    return gradient_history


def plot_result():
    # peer_pressure, academic_pressure, stress_index = read_data()
    x_train,y_train = example_data()
    history = gradient_descent(x_train, y_train, 0, 0, alpha=0.01, num_iters=1000)

    # Get final parameters
    final_w = history[-1]['w']
    final_b = history[-1]['b']

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.5, label='Data points')

    # Plot the fitted line
    x_line = np.linspace(x_train.min(), x_train.max(), 100)
    y_line = final_w * x_line + final_b
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {final_w:.2f}x + {final_b:.2f}')

    plt.xlabel('Peer Pressure')
    plt.ylabel('Stress Index')
    plt.title('Linear Regression: Peer Pressure vs Stress Index')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_result()


