PATIENT_NAME_COL = "patient_name"
CELL_TYPE_COL = "cell_type"
CELL_NAME_COL = "cell_name"
PATIENT_STATE_COL = "patient_state"
CELL_ID_COL = "spectre"

def pre_process_data(df, label_encoder_cell_type, label_encoder_patient_state):

    X_train = df.drop([PATIENT_STATE_COL, PATIENT_NAME_COL, CELL_NAME_COL, CELL_ID_COL], axis=1)
    y_train = df[PATIENT_STATE_COL]

    X_train[CELL_TYPE_COL] = label_encoder_cell_type.transform(X_train[CELL_TYPE_COL])
    
    y_train = label_encoder_patient_state.transform(y_train)

    return X_train, y_train 

def pre_process_data_for_ensembling(df, label_encoder_cell_type, label_encoder_patient_state):

    X_lst = []
    y_lst = []
    for _, group in df.groupby(PATIENT_NAME_COL):
        X, y = pre_process_data(group, label_encoder_cell_type, label_encoder_patient_state)
        X_lst.append(X)
        y_lst.append(y[0])

    return X_lst, y_lst