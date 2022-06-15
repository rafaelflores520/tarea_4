import sys
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def dbs(data, g_eps, min):
    dbs = DBSCAN(eps=g_eps, min_samples=min).fit(data)
    print(dbs.labels_)
    plt.scatter(data[:, 0], data[:, 1], c=dbs.labels_)
    plt.show()

def main(argv):
    if len(argv)==3:
        file = argv[0]
        eps = float(argv[1])
        min_sample = int(argv[2])
        data = pd.read_csv(file, header=0).to_numpy()
        dbs(data, eps, min_sample)
    else:
        print('El formato correcto es: python3 dbscan.py <file> <eps> <min_sample>')
    print("Completed.")

if __name__ == "__main__":
    main(sys.argv[1:]) 