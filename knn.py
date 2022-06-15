from ast import arg
import sys
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import re

def natural_to_number(val):
    if (val == "Si"):
        return 1
    elif (val == "No"):
        return 0

conv_dict = {
    "acidos": natural_to_number,
    "aire": natural_to_number,
    "amargos": natural_to_number,
    "arroz": natural_to_number,
    "cerdo": natural_to_number,
    "dulces": natural_to_number,
    "exotica": natural_to_number,
    "frito": natural_to_number,
    "jugosa": natural_to_number,
    "mariscos": natural_to_number,
    "picante": natural_to_number,
    "salada": natural_to_number,
    "sopas": natural_to_number,
    "te": natural_to_number,
    "vegetariana": natural_to_number,
}

def knn (data, k ):
    features = data.columns.to_list()[:-1]
    X = data[features]
    Y = data["class"]
    neigh = KNeighborsClassifier(n_neighbors=k)
    print('Vamos a entrenar el knn')
    start = time.time()
    neigh = neigh.fit(X, Y)
    end = time.time()
    print((end-start))
    return (neigh,(end-start))

def main(argv):
    if len(argv)==2:
        file =argv[0]
        k = int(argv[1]) 
        data = pd.read_csv(file, header=0, converters=conv_dict)
        neigh = knn(data,k)
        res = re.findall(r'[a-z]+\.|very_[a-z]+\.', file)[0][:-1]
        infotuple = (neigh[0],neigh[1],res)
        dump(infotuple, res+'_knn_k-{k}.joblib'.format(k=k))
    else:
        print('El formato correcto es: python3 knn.py <archivo de dataset.csv> <k que es la cantidad de vecinos>')
 

if __name__ == "__main__":
    main(sys.argv[1:]) 