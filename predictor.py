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

FILEPATH = "raw_data.csv"
FIGURES_DIR = "figures"

PATIENT_NAME_COL = "patient_name"
CELL_TYPE_COL = "cell_type"
CELL_NAME_COL = "cell_name"
PATIENT_STATE_COL = "patient_state"

TEST_SIZE = 0.2

# for reproducibility
RANDOM_STATE = 53

def pre_process_data(df, label_encoder_cell_type, label_encoder_patient_state):

    X_train = df.drop([PATIENT_STATE_COL, PATIENT_NAME_COL, CELL_NAME_COL], axis=1)
    y_train = df[PATIENT_STATE_COL]

    X_train[CELL_TYPE_COL] = label_encoder_cell_type.transform(X_train[CELL_TYPE_COL])
    
    y_train = label_encoder_patient_state.transform(y_train)

    return X_train, y_train 

def pre_process_data_for_ensembling(df, feat_scaler, label_encoder_cell_type, label_encoder_patient_state):

    X_lst = []
    y_lst = []
    for _, group in df.groupby(PATIENT_NAME_COL):
        X, y = pre_process_data(group, label_encoder_cell_type, label_encoder_patient_state)
        X_lst.append(feat_scaler.transform(X))
        y_lst.append(y[0])

    return X_lst, y_lst

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
    X_train, y_train = pre_process_data(train_df, label_encoder_cell_type, label_encoder_patient_state)
    X_val, y_val = pre_process_data(val_df, label_encoder_cell_type, label_encoder_patient_state)

    # scale data
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train)

    scaler_val = StandardScaler()
    X_val_scaled = scaler_val.fit_transform(X_val)

    # fit  linear model
    base_model = LogisticRegression(max_iter=500)
    ensemble_model = SimpleAveragingClassifier(base_model)
    ensemble_model.fit(X_train_scaled, y_train)

    # evaluate model
    X_val_ens, y_val_ens = pre_process_data_for_ensembling(val_df, scaler_val, label_encoder_cell_type, label_encoder_patient_state)
    y_pred_val_proba = ensemble_model.predict_proba(X_val_ens)[:, 1]

    plt.hist(y_pred_val_proba, bins=20)
    plt.ylabel("count")
    plt.xlabel("y pred val prob")
    plt.title("y pred val prob")
    plt.grid("on")
    plt.savefig(os.path.join(FIGURES_DIR, "y_pred_val_proba.png"))
    plt.close()

    # Convert probabilities to binary labels using a threshold of 0.5
    acc = accuracy_score(y_val_ens, (y_pred_val_proba >= 0.5).astype(int))
    print(f"Accuracy: {acc}:")

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_val_ens, y_pred_val_proba)

    # Compute AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    print(f"AUC={roc_auc}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (no discrimination)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, "y_roc_curve_ens.png"))
    plt.close()

    # Optionally, print the model coefficients and intercept
    #print("Model coefficients:", model.coef_)
    print("Model intercept:", ensemble_model.model.intercept_)

    coefficients = np.abs(ensemble_model.model.coef_[0])

    # Sort coefficients in decreasing order
    sorted_indices = np.argsort(coefficients)[::-1]  # Reverse to get descending order

    # Get sorted coefficients
    sorted_coefficients = coefficients[sorted_indices]

    # Get sorted feature names
    sorted_feature_names = np.array(X_train.columns)[sorted_indices]
    print(sorted_feature_names.shape, sorted_coefficients.shape)

    # Plot the sorted coefficients
    plt.barh(sorted_feature_names[:10], sorted_coefficients[:10])
    plt.xlabel('Coefficient Value')
    plt.title('Model Coefficients (Ordered by Decreasing Magnitude)')
    plt.gca().invert_yaxis()  # To display the highest coefficient at the top
    plt.savefig(os.path.join(FIGURES_DIR, "model_coefs.png"))
    plt.close()


if __name__ == "__main__":
    main()