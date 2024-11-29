import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

FILEPATH = "raw_data.csv"
FIGURES_DIR = "figures"

PATIENT_NAME_COL = "patient_name"
CELL_TYPE_COL = "cell_type"
CELL_NAME_COL = "cell_name"
PATIENT_STATE_COL = "patient_state"

def main():
    
    df = pd.read_csv(FILEPATH, delimiter=";")

    # basic data analysis
    print(f"df columns: {df.columns}")
    print(f"df rows: {len(df)}")

    print(f"number of patients: {len(set(df[PATIENT_NAME_COL]))}")
    print(f"number of cell types: {len(set(df[CELL_TYPE_COL]))}")
    print(f"number of cell names: {len(set(df[CELL_NAME_COL]))}")
    print(f"number of patient states: {len(set(df[PATIENT_STATE_COL]))}")
    print(f"number of rows with nan: {np.sum(df.isna().any())}")
    
    # histograms
    plt.hist(df[PATIENT_STATE_COL])
    plt.ylabel("count")
    plt.xlabel("patient state")
    plt.title("patient state histogram")
    plt.grid("on")
    plt.savefig(os.path.join(FIGURES_DIR, "patient_state_hist.png"))
    plt.close()

    label_encoder_cell_name = LabelEncoder()
    df[CELL_NAME_COL+"_encoded"] = label_encoder_cell_name.fit_transform(df[CELL_NAME_COL])
    plt.hist(list(df[CELL_NAME_COL+"_encoded"]), bins=len(label_encoder_cell_name.classes_))
    plt.ylabel("count")
    plt.xlabel("cell name")
    plt.title("cell name histogram")
    plt.grid("on")
    plt.savefig(os.path.join(FIGURES_DIR, "cell_name_hist.png"))
    plt.close()

    return 0

    plt.hist(df[CELL_TYPE_COL])
    plt.ylabel("count")
    plt.xlabel("cell type")
    plt.title("cell type histogram")
    plt.grid("on")
    plt.savefig(os.path.join(FIGURES_DIR, "cell_type_hist.png"))
    plt.close()

    for i in range(1, 11):
        plt.hist(df[f"lambda_{i}"], bins=20)
        plt.ylabel("count")
        plt.xlabel(f"lambda {i}")
        plt.title(f"lambda {i} histogram")
        plt.grid("on")
        plt.savefig(os.path.join(FIGURES_DIR, f"lambda_{i}.png"))
        plt.close()

    

if __name__ == "__main__":
    main()