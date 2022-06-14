import sys
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means(data, n_cluster):
    k_pred = KMeans(n_clusters=n_cluster).fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=k_pred)
    plt.show()

def main(argv):
    #Something
    file = argv[0]
    n = int(argv[1])
    data = pd.read_csv(file, header=0).to_numpy()
    k_means(data, n)
    print("Fin.")

if __name__ == "__main__":
    main(sys.argv[1:]) 