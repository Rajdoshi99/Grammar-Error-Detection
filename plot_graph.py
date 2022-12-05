import matplotlib.pyplot as plt


def plot_graph(X, y, x_label=None, y_label=None, title=None):
    plt.plot(X, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
