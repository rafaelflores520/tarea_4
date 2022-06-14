import sys
import pandas as pd

def main(argv):
    #Something
    file = argv[0]
    data = pd.read_csv(file, header=0).to_numpy()
    print(type(data))
    print("Fin.")

if __name__ == "__main__":
    main(sys.argv[1:]) 