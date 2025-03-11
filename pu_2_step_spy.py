import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def pu_learning_SPY(Ps, Ns, train_mat, test_mat, fold, First_Step, classifier="RF"):):
    """
    Perform Positive-Unlabeled (PU) learning using the SPY technique.d (PU) learning using the SPY technique.
    
    Parameters:
    Ps (array): Indices of positive samples.
    Ns (array): Indices of negative samples.f negative samples.
    train_mat (array): Training data matrix.
    test_mat (array): Test data matrix.
    fold (int): Current fold number for cross-validation.er for cross-validation.
    First_Step (str): The first step method to use ("SPY" or "1DNF").r "1DNF").
    classifier (str): The classifier to use ("RF", "XGB", or "NB").e ("RF", "XGB", or "NB").
    
    Returns:
    array: Predicted probabilities for the test data.e test data.
    """
    np.random.seed(0)
    labeled_data_indexes = Ps
    n_spy = round(len(labeled_data_indexes) * 0.1)len(labeled_data_indexes) * 0.1)
    n_CV = 10  # Number of cross-validation foldsfolds

    prob_spymin_list = []

    # Cross-validation for prob_spymin estimationn
    for i in range(n_CV):
        spy_data_indexes = np.random.choice(labeled_data_indexes, n_spy, replace=False)e(labeled_data_indexes, n_spy, replace=False)
        spy = np.zeros(len(train_mat), dtype=np.int32)e=np.int32)
        spy[spy_data_indexes] = 1
        train_ss = np.zeros(len(train_mat), dtype=np.int32)=np.int32)
        train_ss[labeled_data_indexes] = 1    train_ss[labeled_data_indexes] = 1
        new_train_ss = train_ss & (1 - spy)
        
        # Train the classifier
        _clf = XGBClassifier(eval_metric='logloss', random_state=42)m_state=42)
        _clf.fit(train_mat, new_train_ss))
        
        # Predict probabilities    # Predict probabilities
        train_ss_prob = _clf.predict_proba(train_mat)[:, 1]t)[:, 1]
        prob_spymin = train_ss_prob[spy.astype(bool)].min())].min()
        prob_spymin_list.append(prob_spymin)    prob_spymin_list.append(prob_spymin)

    prob_spymin_mean = np.mean(prob_spymin_list)
    prob_spymin_min = np.min(prob_spymin_list)st)
    prob_spymin_selected = prob_spymin_minob_spymin_min

    data_indexes_pos = labeled_data_indexes
    xs_pos = train_mat[data_indexes_pos]indexes_pos]
    ys_pos = np.ones(len(xs_pos), dtype=np.int32)s_pos), dtype=np.int32)
    
    data_indexes_neg = np.where(train_ss_prob < prob_spymin_selected)[0]indexes_neg = np.where(train_ss_prob < prob_spymin_selected)[0]
    xs_neg = train_mat[data_indexes_neg]
    ys_neg = np.zeros(len(xs_neg), dtype=np.int32)eg = np.zeros(len(xs_neg), dtype=np.int32)
    
    new_train_xs = np.concatenate([xs_neg, xs_pos])atenate([xs_neg, xs_pos])
    new_train_ys = np.concatenate([ys_neg, ys_pos])
    
    # Train the final classifierlassifier
    if classifier == 'RF':    if classifier == 'RF':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier == 'XGB':
        clf = XGBClassifier(eval_metric='logloss', random_state=42)sifier(eval_metric='logloss', random_state=42)
    elif classifier == 'NB':
        clf = GaussianNB()
    else:
        raise ValueError("Unsupported classifier type. Choose 'RF', 'XGB', or 'NB'") 'RF', 'XGB', or 'NB'")
    
    clf.fit(new_train_xs, new_train_ys)
    
    # Predict probabilities for the test data
    test_ys_prob = clf.predict_proba(test_mat)[:, 1]
    
    # Save prob_spymin and negative size
    prob_spymin_name = f"/data/JC/PUlearn/Sepsis/PU_py/prob_spymin_{First_Step}_{classifier}_{fold}.txt"d}.txt"
    np.savetxt(prob_spymin_name, [prob_spymin_selected], fmt="%f")_selected], fmt="%f")
    
    neg_size = ys_neg.size
    neg_size_name = f"/data/JC/PUlearn/Sepsis/PU_py/neg_size_{First_Step}_{classifier}_{fold}.txt"{First_Step}_{classifier}_{fold}.txt"
    np.savetxt(neg_size_name, [neg_size], fmt="%d")
    
    return test_ys_prob

def pulearn(cr, fold, pos_folds, positiv_col="HAC", cost_only=True, First_Step="SPY", classifier="RF"):olds, positiv_col="HAC", cost_only=True, First_Step="SPY", classifier="RF"):
    """
    Perform Positive-Unlabeled (PU) learning with specified parameters.ers.
    
    Parameters:
    cr (DataFrame): Input data.
    fold (int): Current fold number for cross-validation.
    pos_folds (array): Array indicating positive folds.ndicating positive folds.
    positiv_col (str): Column name for positive labels.name for positive labels.
    cost_only (bool): Whether to use only cost columns.er to use only cost columns.
    First_Step (str): The first step method to use ("SPY" or "1DNF"). step method to use ("SPY" or "1DNF").
    classifier (str): The classifier to use ("RF", "XGB", or "NB").fier to use ("RF", "XGB", or "NB").
    
    Returns:    Returns:
    dict: Dictionary containing sensitivity and predicted probabilities.
    """
    np.random.seed(1)
    positiv_names = ["HAC", "angus", "angus_inf", "angus_org"]iv_names = ["HAC", "angus", "angus_inf", "angus_org"]
    cr = cr.drop(columns=[col for col in positiv_names if col not in positiv_col], errors='ignore')e')
    
    cost_cols = cr.loc[:, "administration":"transporte"].columnscolumns
    chop_cols = cr.loc[:, "Auge":"Versch_diagn_ther"].columnsther"].columns
    keep_variables = [col for col in ["icd_c", "id_jahr"] + [positiv_col] if col in cr.columns]ol in cr.columns]
    cr = cr[list(cost_cols) + list(chop_cols) + keep_variables]
    
    if "icd_c" in cr.columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        icd_encoded = encoder.fit_transform(cr[["icd_c"]])["icd_c"]])
        icd_encoded_df = pd.DataFrame(icd_encoded, columns=encoder.get_feature_names_out(["icd_c"]))s=encoder.get_feature_names_out(["icd_c"]))
        cr = pd.concat([cr.drop(columns=["icd_c"]), icd_encoded_df], axis=1)
    
    cost = cr[list(cost_cols)].fillna(0).replace("NULL", 0).apply(pd.to_numeric, errors='coerce')cols)].fillna(0).replace("NULL", 0).apply(pd.to_numeric, errors='coerce')
    cost = cost.loc[:, (cost.nunique() >= 8)]cost = cost.loc[:, (cost.nunique() >= 8)]








































    return list_Se_preds    hac_preds_df.to_csv(filename,index=False)    filename = f"/data/JC/PUlearn/Sepsis/PU_py/preds_{First_Step}_{classifier}_{fold}.csv"    list_Se_preds = {"Se": Se, "hac_preds": hac_preds_df}    Se = TP / (TP + FN) if (TP + FN) > 0 else 0    FN = np.sum((pos_test_pred["class_hat"] == 0) & (pos_test_pred["class_orig"] == 1))    TP = np.sum((pos_test_pred["class_hat"] == 1) & (pos_test_pred["class_orig"] == 1))    pos_test_pred["class_orig"] = 1    pos_test_pred["class_hat"] = (pos_test_pred["pred"] > classification_cutoff).astype(int)    pos_test_pred = pd.DataFrame(hac_preds[len(train_mat):], columns=["pred"])    classification_cutoff = train["HAC"].mean()    hac_preds_df = pd.DataFrame(hac_preds, columns=["P"])            raise ValueError(f"Unsupported first step type '{First_Step}'. Choose 'SPY' or '1DNF'.")    else:        hac_preds = pu_learning_methods[First_Step](Ps, Ns, train_mat, test_mat, fold, First_Step,classifier=classifier)    if First_Step in pu_learning_methods:    # Check if the First_Step is valid and call the corresponding function    }        "1DNF": pu_learning_1DNF        "SPY": pu_learning_SPY,    pu_learning_methods = {    Ns = np.where(cls == 0)[0]    Ps = np.where(cls == 1)[0]    cls = train["HAC"].apply(lambda x: 1 if x == 1 else 0).values    train_mat = train.drop(columns=["HAC"], errors='ignore').astype(float).values    test_mat = cost_test.drop(columns=["HAC"], errors='ignore').astype(float).values    cost_test[numeric_cols] = scaler.transform(cost_test[numeric_cols])    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])    numeric_cols = train.select_dtypes(include=[np.number]).columns.drop("HAC", errors='ignore')    scaler = MinMaxScaler()    train = cost[pos_folds != fold]    cost_test = cost.copy()    cost.rename(columns={positiv_col: "HAC"}, inplace=True)        cost = cost.drop(columns=["icd_c_U50-U52 Functional impairment"], errors='ignore')    if "icd_c_U50-U52 Functional impairment" in cost.columns:    cost = cost.fillna(0).replace("NULL", 0)    cost = pd.concat([cost, cr.drop(columns=cost_cols)], axis=1)    cost = pd.concat([cost, cr.drop(columns=cost_cols)], axis=1)
    cost = cost.fillna(0).replace("NULL", 0)





    return list_Se_preds    hac_preds_df.to_csv(filename,index=False)    filename = f"/data/JC/PUlearn/Sepsis/PU_py/preds_{First_Step}_{classifier}_{fold}.csv"
    list_Se_preds = {"Se": Se, "hac_preds": hac_preds_df}    Se = TP / (TP + FN) if (TP + FN) > 0 else 0    if "icd_c_U50-U52 Functional impairment" in cost.columns:

    FN = np.sum((pos_test_pred["class_hat"] == 0) & (pos_test_pred["class_orig"] == 1))

    TP = np.sum((pos_test_pred["class_hat"] == 1) & (pos_test_pred["class_orig"] == 1))


    pos_test_pred["class_orig"] = 1    pos_test_pred["class_hat"] = (pos_test_pred["pred"] > classification_cutoff).astype(int)
    pos_test_pred = pd.DataFrame(hac_preds[len(train_mat):], columns=["pred"])
    classification_cutoff = train["HAC"].mean()


    hac_preds_df = pd.DataFrame(hac_preds, columns=["P"])            raise ValueError(f"Unsupported first step type '{First_Step}'. Choose 'SPY' or '1DNF'.")

    else:


        hac_preds = pu_learning_methods[First_Step](Ps, Ns, train_mat, test_mat, fold, First_Step,classifier=classifier)

    if First_Step in pu_learning_methods:    # Check if the First_Step is valid and call the corresponding function
    }        "1DNF": pu_learning_1DNF
        "SPY": pu_learning_SPY,    pu_learning_methods = {    Ns = np.where(cls == 0)[0]
    Ps = np.where(cls == 1)[0]    cls = train["HAC"].apply(lambda x: 1 if x == 1 else 0).values        cost = cost.drop(columns=["icd_c_U50-U52 Functional impairment"], errors='ignore')

    train_mat = train.drop(columns=["HAC"], errors='ignore').astype(float).values
    test_mat = cost_test.drop(columns=["HAC"], errors='ignore').astype(float).values    cost.rename(columns={positiv_col: "HAC"}, inplace=True)


    cost_test[numeric_cols] = scaler.transform(cost_test[numeric_cols])
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])    numeric_cols = train.select_dtypes(include=[np.number]).columns.drop("HAC", errors='ignore')
    scaler = MinMaxScaler()    cost_test = cost.copy()
    train = cost[pos_folds != fold]