import matplotlib.pyplot as plt
from data_actions import np_data

def visualize_original():
    X_train,y_train = np_data()
    X_features = ['Hours_Worked', 'Sleep_Hours', 'Bugs', 'Deadline_Days', 'Coffee_Cups',
       'Meetings', 'Interruptions', 'Remote_Work']
    fig, ax = plt.subplots(1, 8, figsize=(12, 6), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Stress Level")
    plt.show()

visualize_original()