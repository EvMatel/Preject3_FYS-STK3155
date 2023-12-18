import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sklm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_gradient(y, X, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    gradient = X.T @ (h - y) / m
    return gradient


def logistic_loss(y, X, theta):
    h = sigmoid(X @ theta)
    loss = -np.mean(y * np.log(np.maximum(h, 1e-15)) + (1 - y) * np.log(np.maximum(1 - h, 1e-15)))
    return loss


def split_and_scale_data(X, y, scaler_type, test_size=0.2, random_state=2023, scale=False):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale:
        # Scale the features based on the specified scaler type
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def grid_search(X_train, X_test, y_train, y_test, optimizer, param_grid, algorithm, n_runs=2, scoring=sklm.accuracy_score):
    best_params = None
    best_avg_score = -np.inf

    for _ in range(n_runs):
        scores = []

        for eta in param_grid['eta']:
            for lmd in param_grid['lmd']:
                if algorithm == 'gd':
                    trained_theta = train_logistic_regression(X_train, y_train, optimizer, eta=eta, lmd=lmd)
                elif algorithm == 'sgd':
                    for minibatch_size in param_grid['minibatch_size']:
                        trained_theta = sgd_logistic_regression(X_train, y_train.reshape(-1, 1), optimizer,
                                                                  minibatch_size, eta=eta, lmd=lmd)

                # Make predictions on the test set
                y_pred_binary = np.round(sigmoid(X_test @ trained_theta))

                # Evaluate accuracy
                score = scoring(y_test, y_pred_binary)
                scores.append(score)

                avg_score = np.mean(scores)

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    if algorithm == 'gd':
                        best_params = {'eta': eta, 'lmd': lmd}
                        param = np.asarray([eta, lmd])
                    elif algorithm == 'sgd':
                        best_params = {'eta': eta, 'lmd': lmd, 'minibatch_size': minibatch_size}
                        param = np.asarray([eta, lmd, minibatch_size])

    print("Best parameters found: ", best_params)
    print("Best average test set score over {} runs: {:.4f}".format(n_runs, best_avg_score))

    return best_params, param


def plot_confusion(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(name)
    plt.show()
    sns.set()


def grid_search_and_plot(X_train, X_test, y_train, y_test, param_grid, optimizers, algorithms, n_runs=10):
    best_params_dict = {}

    for optimizer in optimizers:
        for algorithm in algorithms:
            print(f"Grid search for optimizer '{optimizer}' and algorithm '{algorithm}':")
            best_params, param = grid_search(X_train, X_test, y_train, y_test, optimizer, param_grid, algorithm, n_runs)
            best_params_dict[(optimizer, algorithm)] = best_params

            # Plot confusion matrix for the best run of gd and sgd
            if optimizer == 'gd':
                trained_theta = train_logistic_regression(X_train, y_train, optimizer, n_iter=100, eta=param[0], lmd=param[1])
                y_pred_binary = np.round(sigmoid(X_test @ trained_theta))
                plot_confusion(y_test, y_pred_binary)
                print(f"Optimizer: 'gd', Algorithm: {algorithm}, Best Parameters: {best_params}")

            elif optimizer == 'sgd':
                trained_theta = sgd_logistic_regression(X_train, y_train.reshape(-1, 1), optimizer, n_epochs=100, eta=param[0], lmd=param[1], minibatch_size=param[2])
                y_pred_binary = np.round(sigmoid(X_test @ trained_theta))
                plot_confusion(y_test, y_pred_binary)
                print(f"Optimizer: 'sgd', Algorithm: {algorithm}, Best Parameters: {best_params}")

    return best_params_dict


def adagrad_optimizer(y, X, theta, n_iter, eta, delta=1e-7, lmd=0.01):
    past_g = 0.0

    training_grad = grad(logistic_loss, 2)
    for i in range(n_iter):
        gradient = training_grad(y, X, theta)
        past_g += gradient * gradient
        update = (gradient + lmd * theta) * eta / (np.sqrt(past_g) + delta)
        theta -= update

    return theta


def rmsprop_optimizer(y, X, theta, n_iter, eta, rho=0.99, delta=1e-7, lmd=0.01):
    Giter = 0.0

    for i in range(n_iter):
        gradient = logistic_gradient(y, X, theta)
        Giter = rho * Giter + (1 - rho) * gradient * gradient
        update = (gradient + lmd * theta) * eta / (np.sqrt(Giter) + delta)
        theta -= update

    return theta


def adam_optimizer(y, X, theta, n_iter, eta, beta1=0.9, beta2=0.999, delta=1e-7, lmd=0.01):
    iter = 0
    first_moment = 0.0
    second_moment = 0.0

    for i in range(n_iter):
        gradient = logistic_gradient(y, X, theta) + (lmd * theta)
        iter += 1
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
        first_term = first_moment / (1.0 - beta1 ** iter)
        second_term = second_moment / (1.0 - beta2 ** iter)
        update = eta * first_term / (np.sqrt(second_term) + delta)
        theta -= update

    return theta


def train_logistic_regression(X_train, y_train, optimizer, n_iter=100, eta=0.01, lmd=0.01):
    _, n_features = X_train.shape
    theta = 0.01 * np.random.randn(n_features, 1)

    if optimizer == 'rmsprop':
        theta = rmsprop_optimizer(y_train, X_train, theta, n_iter, eta, lmd)
    elif optimizer == 'adagrad':
        theta = adagrad_optimizer(y_train, X_train, theta, n_iter, eta, lmd)
    elif optimizer == 'adam':
        theta = adam_optimizer(y_train, X_train, theta, n_iter, eta, lmd)

    return theta


def S_adagrad_optimizer(y, X, theta, minibatch_size, n_epochs, eta, delta=1e-7, lmd=0.01, clip_gradients=True):
    n = len(X)
    M = minibatch_size
    m = int(n / M)  # number of minibatches
    data_indices = np.arange(len(X))
    past_g = 0.0

    training_grad = grad(logistic_loss, 2)
    for epoch in range(n_epochs):
        for i in range(m):
            choose_datapoints = np.random.choice(data_indices, size=minibatch_size, replace=False)
            xi = X[choose_datapoints]
            yi = y[choose_datapoints]
            gradient = (1.0 / M) * training_grad(yi, xi, theta) + (lmd * theta)
            if clip_gradients:
                gradient = np.clip(gradient, -1, 1)
            past_g += gradient * gradient
            update = gradient * eta / (np.sqrt(past_g) + delta)
            theta -= update

    return theta


def S_rmsprop_optimizer(y, X, theta, minibatch_size, n_epochs, eta, rho=0.99, delta=1e-7, lmd=0.01, clip_gradients=True):
    n = len(X)
    M = minibatch_size
    m = int(n / M)  # number of minibatches
    iter = 0
    data_indices = np.arange(len(X))

    for epoch in range(n_epochs):
        Giter = 0.0
        iter += 1
        for i in range(m):
            choose_datapoints = np.random.choice(data_indices, size=minibatch_size, replace=False)
            xi = X[choose_datapoints]
            yi = y[choose_datapoints]
            gradient = (1.0 / M) * logistic_gradient(yi, xi, theta) + (lmd * theta)
            if clip_gradients:
                gradient = np.clip(gradient, -1, 1)
            Giter = rho * Giter + (1 - rho) * gradient * gradient
            update = gradient * eta / (np.sqrt(Giter) + delta)
            theta -= update

    return theta


def S_adam_optimizer(y, X, theta, minibatch_size, n_epochs, eta, beta1=0.9, beta2=0.999, delta=1e-7, lmd=0.01,
                   clip_gradients=True):
    n = len(y)
    M = minibatch_size
    m = int(n / M)
    iter = 0
    first_moment = 0.0
    second_moment = 0.0
    data_indices = np.arange(len(X))

    for epoch in range(n_epochs):
        iter += 1
        for i in range(m):
            choose_datapoints = np.random.choice(data_indices, size=minibatch_size, replace=False)
            xi = X[choose_datapoints]
            yi = y[choose_datapoints]
            gradient = (1.0 / M) * logistic_gradient(yi, xi, theta) + (lmd * theta)
            if clip_gradients:
                gradient = np.clip(gradient, -1, 1)
            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
            first_term = first_moment / (1.0 - beta1 ** iter)
            second_term = second_moment / (1.0 - beta2 ** iter)
            update = eta * first_term / (np.sqrt(second_term) + delta)
            theta -= update

    return theta


def sgd_logistic_regression(X_train, Y_train, optimizer, minibatch_size, n_epochs=100, eta=0.01, lmd=0.01):
    _, n_features = X_train.shape
    theta = 0.01 * np.random.randn(n_features, 1)

    if optimizer == 'rmsprop':
        theta = S_rmsprop_optimizer(Y_train, X_train, theta, minibatch_size, n_epochs, eta, lmd)
    elif optimizer == 'adagrad':
        theta = S_adagrad_optimizer(Y_train, X_train, theta, minibatch_size, n_epochs, eta, lmd)
    elif optimizer == 'adam':
        theta = S_adam_optimizer(Y_train, X_train, theta, minibatch_size, n_epochs, eta, lmd)

    return theta


def RandomForestAnalysis(X_train, y_train, X_test, y_test, n_estimators, n_iter=20):
    mse_train_array = np.zeros((n_iter, len(n_estimators)))
    mse_test_array = np.zeros((n_iter, len(n_estimators)))
    r2_train_array = np.zeros((n_iter, len(n_estimators)))
    r2_test_array = np.zeros((n_iter, len(n_estimators)))

    for i in range(n_iter):
        for j in range(len(n_estimators)):

            # Create a Random Forest Regressor
            rf_regressor = RandomForestRegressor(n_estimators=n_estimators[j], random_state=2023)

            # Fit the model
            rf_regressor.fit(X_train, y_train)

            # Make predictions
            y_train_pred = rf_regressor.predict(X_train)
            y_test_pred = rf_regressor.predict(X_test)

            # Evaluate the model
            mse_train = mean_squared_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)

            mse_train_array[i, j] = mse_train
            r2_train_array[i, j] = r2_train
            mse_test_array[i, j] = mse_test
            r2_test_array[i, j] = r2_test

    avg_mse_train = np.mean(mse_train_array, axis=0)
    avg_r2_train = np.mean(r2_train_array, axis=0)
    avg_mse_test = np.mean(mse_test_array, axis=0)
    avg_r2_test = np.mean(r2_test_array, axis=0)

    print(f'Average Random Forest MSETrain={avg_mse_train}')
    print(f'Average Random Forest R2Train={avg_r2_train}')
    print(f'Average Random Forest MSETest={avg_mse_test}')
    print(f'Average Random Forest R2Test={avg_r2_test}')



if __name__ == "__main__":
    np.random.seed(2023)

    spambase = fetch_ucirepo(id=94)

    # data (as pandas dataframes)
    X = np.asarray(spambase.data.features)
    y = np.asarray(spambase.data.targets)

    # Choose scaling: 'minmax', 'standard' or False

    X_train, X_test, y_train, y_test = split_and_scale_data(X, y, scale=True, scaler_type='standard')
    """
    param_grid = {
        'eta': [0.001, 0.01, 0.0001, 0.1],
        'lmd': [0.0001, 0.001, 0.01, 0.1],
        'minibatch_size': [32, 64, 128]
    }


    optimizers = ['adagrad', 'rmsprop', 'adam']
    algorithms = ['gd', 'sgd']

    # Choose an optimizer: 'rmsprop', 'adagrad', or 'adam'
    optimizer = 'adam'
    algorithm = 'gd'

    # b = grid_search(X_train, X_test, y_train, y_test, optimizer, param_grid, algorithm, n_runs=2, scoring=sklm.accuracy_score)
    # print(b)
    # grid_search_and_plot(X_train, X_test, y_train, y_test, param_grid, optimizers, algorithms)


    trained_theta = train_logistic_regression(X_train, y_train, optimizer, n_iter=100, eta=0.01, lmd=0.1)
    y_pred_binary = np.round(sigmoid(X_test @ trained_theta))
    
    accuracy = sklm.accuracy_score(y_test, y_pred_binary)
    print(f"Testing Accuracy: {accuracy}")
    
    minibatch_size = 128
    trained_theta_S = sgd_logistic_regression(X_train, y_train.reshape(-1, 1), optimizer, minibatch_size, n_epochs=100, eta=0.001, lmd=0.0001)
    # Make predictions on the test set
    y_pred_binary_S = np.round(sigmoid(X_test @ trained_theta_S))

    
    # Evaluate accuracy
    S_accuracy = sklm.accuracy_score(y_test, y_pred_binary_S)
    print(f"Testing Accuracy: {S_accuracy}")
    
    plot_confusion(y_test, y_pred_binary, name='gd')

    
    print("Grid search for best parameters:")
    grid_search(X, y, param_grid=param_grid, optimizer='adagrad', algorithm='sgd')
    

    avg_acc = average_accuracy(X_train, X_test, y_train, y_test, optimizer, 'sgd', n_iter=50, eta=0.01, lmd=0.001, minibatch_size=128)
    print(f"Average Accuracy: {avg_acc}")

    trained_theta = train_logistic_regression(X_train, y_train, optimizer, n_iter=100, eta=0.01)
    y_pred_binary = np.round(sigmoid(X_test @ trained_theta))
    plot_confusion(y_test, y_pred_binary)

    y_pred = RandomForestAnalysis(X_train, y_train, X_test, y_test, n_estimators=[50, 100, 150], n_iter=20)
    """


