import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, accuracy_score, auc
import numpy as np
from ensemble_classifier import SimpleAveragingClassifier
from data_preproc import pre_process_data_for_ensembling, PATIENT_NAME_COL, CELL_TYPE_COL, CELL_NAME_COL, PATIENT_STATE_COL

FILEPATH = "raw_data.csv"
FIGURES_DIR = "figures"

TEST_SIZE = 0.2

# for reproducibility
RANDOM_STATE = 53

def plot_roc_curve(y, y_pred, fname="y_roc_curve_ens.png"):
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    # Compute AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (no discrimination)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, fname))
    plt.close()

def main():

    df = pd.read_csv(FILEPATH, delimiter=";")

    df_grouped_patient = [group for _, group in df.groupby(PATIENT_NAME_COL)]
    df_grouped_patient_state = [group[PATIENT_STATE_COL].iloc[0] for group in df_grouped_patient]

    label_encoder_patient_state = LabelEncoder()
    patient_state_encoded = label_encoder_patient_state.fit_transform(df_grouped_patient_state)
    print(patient_state_encoded)

    train, val = train_test_split(
        df_grouped_patient, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=patient_state_encoded
        )

    train_df = pd.concat(train, axis=0)
    val_df = pd.concat(val, axis=0)
    print(val_df)

    print(f"train df, rows: {len(train_df)}, patients {len(set(train_df[PATIENT_NAME_COL]))}")
    print(f"val df, rows: {len(val_df)}, patients {len(set(val_df[PATIENT_NAME_COL]))}")

    # prepare data for training
    label_encoder_cell_type = LabelEncoder()
    label_encoder_cell_type.fit(train_df[CELL_TYPE_COL])
    label_encoder_patient_state = LabelEncoder()
    label_encoder_patient_state.fit(train_df[PATIENT_STATE_COL])
    label_encoder_patient_state.classes_ = ['sain', 'malade'] # force 'sain' to be class 0, and 'malade' to be class 1
    X_train, y_train = pre_process_data_for_ensembling(train_df, label_encoder_cell_type, label_encoder_patient_state)


    # fit  linear model
    base_model = LogisticRegression(max_iter=500)
    ensemble_model = SimpleAveragingClassifier(base_model, scale_data=True)
    ensemble_model.fit(X_train, y_train)

    # evaluate model
    X_val_ens, y_val_ens = pre_process_data_for_ensembling(val_df, label_encoder_cell_type, label_encoder_patient_state)
    y_pred_val_proba = ensemble_model.predict_proba(X_val_ens)[:, 1]
    y_pred_val_proba_wo_ens = ensemble_model.predict_proba_wo_ensembling(X_val_ens)[:, 1]
    y_val = np.concat([[label]*len(df) for df,label in zip(X_val_ens,y_val_ens)])

    plt.hist(y_pred_val_proba, bins=20)
    plt.ylabel("count")
    plt.xlabel("y pred val prob")
    plt.title("y pred val prob")
    plt.grid("on")
    plt.savefig(os.path.join(FIGURES_DIR, "y_pred_val_proba.png"))
    plt.close()

    plt.hist(y_pred_val_proba_wo_ens, bins=20)
    plt.ylabel("count")
    plt.xlabel("y pred val prob w/o ensembling")
    plt.title("y pred val prob w/o ensembling")
    plt.grid("on")
    plt.savefig(os.path.join(FIGURES_DIR, "y_pred_val_proba_wo_ens.png"))
    plt.close()

    # Convert probabilities to binary labels using a threshold of 0.5
    acc = accuracy_score(y_val_ens, (y_pred_val_proba >= 0.5).astype(int))
    print(f"Accuracy: {acc}:")

    plot_roc_curve(y_val_ens, y_pred_val_proba, fname="y_roc_curve_ens.png")
    plot_roc_curve(y_val, y_pred_val_proba_wo_ens, fname="y_roc_curve.png")


    # Plot the model coefficients and print intercept
    print("Model intercept:", ensemble_model.model.intercept_)

    coefficients = np.abs(ensemble_model.model.coef_[0])

    # Sort coefficients in decreasing order
    sorted_indices = np.argsort(coefficients)[::-1]  # Reverse to get descending order

    # Get sorted coefficients
    sorted_coefficients = coefficients[sorted_indices]
    sorted_coefficients = sorted_coefficients / np.linalg.norm(sorted_coefficients)

    # Get sorted feature names
    sorted_feature_names = np.array(X_train[0].columns)[sorted_indices]

    # Plot the sorted coefficients
    plt.barh(sorted_feature_names[:15], sorted_coefficients[:15])
    plt.xlabel('Normalized Value')
    plt.title('Normalized model Coefficients (Ordered by Decreasing Magnitude)')
    plt.gca().invert_yaxis()  # To display the highest coefficient at the top
    plt.grid()
    plt.savefig(os.path.join(FIGURES_DIR, "model_coefs.png"))
    plt.close()


if __name__ == "__main__":
    main()