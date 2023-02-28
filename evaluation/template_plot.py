from matplotlib import pyplot as plt
import csv
import numpy as np
import pandas as pd
import seaborn as sns


def get_data(filename):
    df = pd.read_csv(filename)
    return df


def heatmap_plot(title):
    plt.title(title)
    a = np.random.random((16, 16))
    plt.imshow(a, cmap="hot", interpolation="nearest")
    plt.show()


def box_plot(title):
    plt.title(title)
    data_1 = np.random.normal(100, 10, 200)
    data_2 = np.random.normal(90, 20, 200)
    data_3 = np.random.normal(80, 30, 200)
    data_4 = np.random.normal(70, 40, 200)
    data = [data_1, data_2, data_3, data_4]
    plt.boxplot(data)
    plt.show()


def plot_samples_b_8_test():
    df = get_data("maze_samples_b_8.csv")
    # Erstellen des Liniendiagramms
    sns.set_style("darkgrid")
    sns.set_palette("bright")

    plt.figure(figsize=(8, 6))
    # sns.lineplot(data=df, x="complexity", y="A*")

    # Hinzufügen einer nicht-linearen Regression
    sns.regplot(data=df, x="complexity", y="algorithm", order=2, ci=None, scatter=False, label="Regression")

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title("Algorithmic Performance Comparison over Time")
    plt.xlabel("Complexity")
    plt.ylabel("Execution Time (ms)")

    # Anzeige des Diagramms
    plt.show()


def line_plot_samples(title, data):

    plt.title(title)
    sns.lineplot(x="complexity", y="samples", hue="algorithm", data=data)
    plt.show()


def line_plot_samples_reward(title, data):
    plt.title(title)
    sns.lineplot(x="samples", y="reward", hue="algorithm", data=data)
    plt.show()


"""plot samples of A*, MCTS, LAP3 with b=4"""


def plot_samples_b_4():

    data = get_data("maze_samples_b_4.csv")
    line_plot_samples("maze samples with b=4", data)


"""plot reward of A*, MCTS, LAP3 with b=4"""


def plot_reward_b_4():
    data = get_data("maze_samples_b_4.csv")
    line_plot_samples("maze samples with b=4", data)


"""plot samples of A*, MCTS, LAP3 with b=8"""


def plot_samples_b_8():
    data = get_data("maze_samples_b_8.csv")

    line_plot_samples("maze samples with b=8", data)


def plot_reward_b_8():
    data = get_data("maze_samples_b_8.csv")
    # x = data["complexity"].values
    y = data[["complexity"]].values
    astar = data[["A*"]].values
    mcts = data[["MCTS"]]
    lap3 = data[["LAP3"]]
    line_plot_samples("maze samples with b=8", data)


"""plot samples of A*, MCTS, LAP3 with b=16"""


def plot_samples_b_16():
    data = get_data("maze_samples_b_16.csv")

    line_plot_samples("maze samples with b=16", data)


def plot_samples_b_8_samples_reward_list():
    data = get_data("maze_samples_b_8_samples_reward_list.csv")

    line_plot_samples_reward("plot_samples_b_8_samples_reward_list", data)


def main():

    # plot_samples_b_4()
    # plot_samples_b_8()
    # plot_reward_b_8()
    # plot_samples_b_16()
    # box_plot("test")
    # heatmap_plot("test")
    # plot_samples_b_8_test()
    # plot_samples_b_8_test()
    # plot_samples_b_8_with_noise()
    plot_samples_b_8_samples_reward_list()


if __name__ == "__main__":
    main()
