import numpy as np
import matplotlib.pyplot as plt


def plot_loss_per_minibatch(loss, train_errors, validate_errors, test_errors, save_fig=False):
    x_values_train = np.arange(1, len(train_errors) + 1)
    x_values_val = np.arange(len(train_errors) // len(validate_errors), len(train_errors) + 1, len(train_errors) // len(validate_errors))
    rest = len(train_errors) % len(validate_errors)
    if rest != 0:
        x_values_val = np.append(x_values_val, x_values_val[-1] + rest)
    x_values_test = np.arange(len(train_errors) + 1, len(train_errors) + 61)
    test_errors = np.repeat(test_errors[0], 60)

    plt.figure(100)
    plt.plot(x_values_train, train_errors, label="Train")
    plt.plot(x_values_val, validate_errors, label="Validate")
    plt.plot(x_values_test, test_errors, label="Test")

    plt.xlabel("Minibatch")
    plt.ylabel(loss)
    title = get_loss_string(loss) + ' per minibatch'
    plt.title(title)
    plt.legend(fontsize="large")
    if save_fig:
        plt.savefig('data/result/fig-loss', bbox_inches='tight')
    plt.show()


def get_loss_string(loss):
    if loss == 'cross_entropy':
        return "Cross-entropy"
    if loss == 'mse':
        return 'MSE'
    return ''
