import sys
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def agg_clustering(data, n_cluster=None, distance=None):
    agg = AgglomerativeClustering(n_clusters=n_cluster, distance_threshold=distance).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=agg.labels_)
    plt.show()

def main(argv):
    #Something
    file = argv[0]
    data = pd.read_csv(file, header=0).to_numpy()
    opts = [opt for opt in argv[1:] if opt.startswith("-")]
    args = [arg for arg in argv[1:] if not arg.startswith("-")]
    if ("-n_cluster" in opts):
        n = int(args[0])
        agg_clustering(data, n_cluster=n)
    elif ("-distance" in opts):
        d = int(args[0])
        agg_clustering(data, distance=d)
    else:
        print('El formato correcto es:\n python3 agglomerative_clustering.py <file> -n_cluster <int> \nor\n')
        print('python3 agglomerative_clustering.py <file> -distance <int>')
    print("Completed.")

if __name__ == "__main__":
    main(sys.argv[1:]) 