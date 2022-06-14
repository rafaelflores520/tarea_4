import sys
import time
import pandas as pd
from joblib import load
from sklearn import metrics

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

def parse_data(raw):
    return pd.read_csv(raw, header=0, converters=conv_dict)

def metrics_pack(y_pred, y_true, clf, tr_time, val_time):
    val_acc = metrics.accuracy_score(y_true, y_pred)
    val_rec = metrics.recall_score(y_true, y_pred, average="macro")
    val_pre = metrics.precision_score(y_true, y_pred, average="macro")
    val_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=clf.classes_)

    print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(val_acc, val_rec, val_pre, val_f1, tr_time, val_time))
    df = pd.DataFrame(matrix, columns = clf.classes_)
    print("Mean Acc.:\t{:.4f}\nMean Rec.:\t{:.4f}\nMean Prec.:\t{:.4f}\nF1 Score:\t{:.4f}\nTr. Time:\t{}\nVal. Time:\t{}".format(val_acc, val_rec, val_pre, val_f1, tr_time, val_time))
    print(df)

def predict(model, validation):
    data = parse_data(validation)
    features = data.columns.to_list()[:-1]
    X = data[features]
    Y = data["class"]

    #Cargando el modelo
    model_info = load(model) 
    clf = model_info[0]
    start = time.time()
    Y_pred = clf.predict(X)
    end = time.time()
    val_time = end - start
    metrics_pack(Y_pred, Y, clf, model_info[1], val_time)


def main(argv):
    if (len(argv) != 2):
        print("Puede que le falten argumento o tenga de mas /n Argumentos: <archivo_modelo> <archivo_validacion>")
        return 0
    predict(argv[0], argv[1])



if __name__ == "__main__":
    main(sys.argv[1:]) 