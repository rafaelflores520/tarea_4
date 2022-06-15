import sys
import pandas as pd
import matplotlib.pyplot as plt

def print(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.legend()
    plt.show()

def main(argv):
        file = argv[0]
        data = pd.read_csv(file, header=0).to_numpy()
        print(data)

if __name__ == "__main__":
    main(sys.argv[1:]) 