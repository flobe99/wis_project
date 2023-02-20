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


def test():
    df = get_data("test.csv")
    # Erstellen des Liniendiagramms
    sns.set_style("darkgrid")
    sns.set_palette("bright")

    plt.figure(figsize=(8, 6))
    # sns.lineplot(data=df, x="complexity", y="A*")

    # Hinzufügen einer nicht-linearen Regression
    sns.regplot(data=df, x="complexity", y="A*", order=2, ci=None, scatter=False, label="Regression")

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title("Algorithmic Performance Comparison over Time")
    plt.xlabel("Complexity")
    plt.ylabel("Execution Time (ms)")

    # Anzeige des Diagramms
    plt.show()


def line_plot_samples(title, y, astar, mcts, lap3):

    plt.title(title)
    plt.plot(y, astar, color="red")
    plt.plot(y, mcts, color="blue")
    plt.plot(y, lap3, color="green")
    plt.ylabel("samples")
    plt.xlabel("space complexity")
    plt.show()


"""plot samples of A*, MCTS, LAP3 with b=4"""


def plot_samples_b_4():

    data = get_data("maze_samples_b_4.csv")
    # x = data["complexity"].values
    y = data[["complexity"]].values
    astar = data[["A*"]].values
    mcts = data[["MCTS"]]
    lap3 = data[["LAP3"]]
    line_plot_samples("maze samples with b=4", y, astar, mcts, lap3)


"""plot reward of A*, MCTS, LAP3 with b=4"""


def plot_reward_b_4():
    data = get_data("maze_samples_b_4.csv")
    # x = data["complexity"].values
    y = data[["complexity"]].values
    astar = data[["A*"]].values
    mcts = data[["MCTS"]]
    lap3 = data[["LAP3"]]
    line_plot_samples("maze samples with b=4", y, astar, mcts, lap3)


"""plot samples of A*, MCTS, LAP3 with b=8"""


def plot_samples_b_8():
    x = [12, 24, 36, 48, 60]
    astar = [50, 100, 200, 300, 500]
    mcts = [50, 100, 150, 200, 250]
    lap3 = [50, 75, 87, 93, 99]

    line_plot_samples("maze samples with b=8", x, astar, mcts, lap3)


"""plot samples of A*, MCTS, LAP3 with b=16"""


def plot_samples_b_16():
    x = [12, 24, 36, 48, 60]
    astar = [50, 100, 200, 300, 500]
    mcts = [50, 100, 150, 200, 250]
    lap3 = [50, 75, 87, 93, 99]

    line_plot_samples("maze samples with b=16", x, astar, mcts, lap3)


def main():

    plot_samples_b_4()
    plot_samples_b_8()
    plot_samples_b_16()
    box_plot("test")
    heatmap_plot("test")

    test()


if __name__ == "__main__":
    main()
