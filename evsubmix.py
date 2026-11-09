import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
import argparse,pickle
import multi.modules.evaluatesub as es
from multi.modules.models import get_model

def train_test(clf, X_train, y_train, X_test_ti, y_test_ti):
    clf.fit(X_train, y_train)
    sall = []
    for tt in range(10):
        score_func1 = "decision_function" if hasattr(clf, "decision_function") else "predict_proba"
        s1 = es.func_map[score_func1](clf, X_test_ti[tt])
        sall.append(s1)
    score_m = np.mean(np.array(sall), axis=0) 
    y_pred_r = (score_m >= 0).astype(int) if score_m.ndim == 1 else np.argmax(score_m, axis=1)
    cm = confusion_matrix(y_test_ti, y_pred_r, normalize='true')
    return cm

def evaluate_all_samples(ss, data, clf2, clf4, n_splits=10):
    X2, y2, X4, y4 = data[ss]
    cm2, cm4 = [], []
    split_folds2 = es.split_fold(X2, y2, n_splits=n_splits)
    for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in split_folds2:
        clf_fold = clone(clf2)
        cm = train_test(clf_fold, X_train, y_train, X_test_ti, y_test_ti)
        cm2.append(cm)
    split_folds4 = es.split_fold(X4, y4, n_splits=n_splits)
    for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in split_folds4:
        clf_fold = clone(clf4)
        cm = train_test(clf_fold, X_train, y_train, X_test_ti, y_test_ti)
        cm4.append(cm)
    return np.array(cm2), np.array(cm4)

def evaluate_cross_subject(ss, data, clf2, clf4):
    train = [data[i] for i in range(len(data)) if i != ss]
    X_train2 = np.concatenate([d[0] for d in train])[:,:,:150]
    y_train2 = np.concatenate([d[1] for d in train])
    X_train4 = np.concatenate([d[2] for d in train])[:,:,:150]
    y_train4 = np.concatenate([d[3] for d in train])
    X_test2, y_test2, X_test4, y_test4 = data[ss]
    # prepare testing data for imagery
    X_test_no2 = np.concatenate([X_test2[:, :, 750+j*25:900+j*25] for j in range(25)])
    X_test_no4 = np.concatenate([X_test4[:, :, 750+j*25:900+j*25] for j in range(25)])
    y_test_no2 = np.tile(y_test2, 25)
    y_test_no4 = np.tile(y_test4, 25)
    # prepare testing data for imagery with time intervals
    X_test_ti2 = np.concatenate([[X_test2[:, :, 750+(j+t)*25:900+(j+t)*25] for t in range(10)] for j in range(16)], axis=1)
    X_test_ti4 = np.concatenate([[X_test4[:, :, 750+(j+t)*25:900+(j+t)*25] for t in range(10)] for j in range(16)], axis=1)
    y_test_ti2 = np.tile(y_test2, 16)
    y_test_ti4 = np.tile(y_test4, 16)
    # train and test
    clf2_clone = clone(clf2)
    clf4_clone = clone(clf4)
    cm2 = train_test(clf2_clone, X_train2, y_train2, X_test_ti2, y_test_ti2)
    cm4 = train_test(clf4_clone, X_train4, y_train4, X_test_ti4, y_test_ti4)
    return cm2, cm4

def evaluate_in(ss, data, clf2, clf4, n_splits=10):
    X2, y2, X4, y4 = data[ss]
    dp2, di2 = es.split_foldin(X2, y2, n_splits=n_splits)
    cm2p = []
    for X_train_p, y_train_p, X_test_p, y_test_p in dp2:
        clf_fold2 = clone(clf2)
        clf_fold2.fit(X_train_p, y_train_p)
        y_pred_p = clf_fold2.predict(X_test_p)
        cm2p.append(confusion_matrix(y_test_p, y_pred_p, normalize='true'))
    cm2i = []
    for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in di2:
        clf_fold2 = clone(clf2)
        cm2i_flod = []
        for jj in range(25):
            cm = train_test(clf_fold2, X_train[jj], y_train[jj], X_test_ti, y_test_ti)
            cm2i_flod.append(cm)
        cm2i.append(cm2i_flod)
    # 4 class
    dp4, di4 = es.split_foldin(X4, y4, n_splits=n_splits)
    cm4p = []
    for X_train_p, y_train_p, X_test_p, y_test_p in dp4:
        clf_fold4 = clone(clf4)
        clf_fold4.fit(X_train_p, y_train_p)
        y_pred_p = clf_fold4.predict(X_test_p)
        cm4p.append(confusion_matrix(y_test_p, y_pred_p, normalize='true'))
    cm4i = []
    for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in di4:
        clf_fold4 = clone(clf4)
        cm4i_flod = []
        for jj in range(25):
            cm = train_test(clf_fold4, X_train[jj], y_train[jj], X_test_ti, y_test_ti)
            cm4i_flod.append(cm)
        cm4i.append(cm4i_flod)
    return np.array(cm2p), np.array(cm4p), np.array(cm2i), np.array(cm4i)

def evaluate_cross_subject_in(ss, data, clf2, clf4):
    # prepare training and testing data
    train = [data[i] for i in range(len(data)) if i != ss]
    X_train2 = np.concatenate([d[0] for d in train])
    y_train2 = np.concatenate([d[1] for d in train])
    X_train4 = np.concatenate([d[2] for d in train])
    y_train4 = np.concatenate([d[3] for d in train])
    X_test2, y_test2, X_test4, y_test4 = data[ss]
    # prepare training and testing data for perception
    X_train2_p = X_train2[:,:,:150]
    X_train4_p = X_train4[:,:,:150]
    X_test2_p = X_test2[:,:,:150]
    X_test4_p = X_test4[:,:,:150]
    # prepare training and testing data for imagery
    X_train2_i = [X_train2[:, :, 750+j*25:900+j*25] for j in range(25)]
    X_train4_i = [X_train4[:, :, 750+j*25:900+j*25] for j in range(25)]
    y_train2_i = y_train2
    y_train4_i = y_train4
        
    X_test2_i_no  = np.concatenate([X_test2[:, :, 750+j*25:900+j*25] for j in range(25)])
    X_test4_i_no  = np.concatenate([X_test4[:, :, 750+j*25:900+j*25] for j in range(25)])
    y_test2_i_no  = np.tile(y_test2, 25)
    y_test4_i_no  = np.tile(y_test4, 25)
            
    X_test2_i_ti =  np.concatenate([[X_test2[:, :, 750+(j+t)*25:900+(j+t)*25] for t in range(10)] for j in range(16) ], axis=1)
    X_test4_i_ti =  np.concatenate([[X_test4[:, :, 750+(j+t)*25:900+(j+t)*25] for t in range(10)] for j in range(16) ], axis=1)
    y_test2_i_ti = np.tile(y_test2, 16)
    y_test4_i_ti = np.tile(y_test4, 16)
    # train and test
    clf2_clone = clone(clf2)
    clf4_clone = clone(clf4)
    clf2_clone.fit(X_train2_p, y_train2)
    clf4_clone.fit(X_train4_p, y_train4)
    
    y_pred_p2 = clf2_clone.predict(X_test2_p)
    cm2p = confusion_matrix(X_test2_p, y_pred_p2, normalize='true')
    y_pred_p4 = clf4_clone.predict(X_test4_p)
    cm4p = confusion_matrix(X_test4_p, y_pred_p4, normalize='true')
    cm2i, cm4i = [],[]
    for jj in range(25):
        clf2_clone = clone(clf2)
        clf4_clone = clone(clf4)
        cm2 = train_test(clf2_clone, X_train2_i[jj], y_train2_i, X_test2_i_ti, y_test2_i_ti)
        cm4 = train_test(clf4_clone,X_train4_i[jj], y_train4_i, X_test4_i_ti, y_test4_i_ti)
        cm2i.append(cm2)
        cm4i.append(cm4)
    return np.array(cm2p), np.array(cm4p), np.array(cm2i), np.array(cm4i)

parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=int, required=True)
parser.add_argument("--clf", type=str, required=True,
    choices=["lr", "xdawn", "eegnet", "eegconf", "atteca", "eegnex"],)
parser.add_argument("--eval", type=str, required=True,
    choices=["ori", "csub", "in", "cin"],)
args = parser.parse_args()

subnum = args.sub
sclf = args.clf
seval = args.eval

eval_dict = {
    "ori": evaluate_all_samples,
    "csub": evaluate_cross_subject,
    "in": evaluate_in,
    "cin": evaluate_cross_subject_in,
}

models = get_model(sclf)

with open("multi/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

spath = f"multi/acc/mix/sub/{sclf}_{seval}_{subnum}"

if seval in ["in", "cin"]:
    cm2p, cm4p, cm2i, cm4i = eval_dict[seval](subnum, train_data,models[0],models[1])
    np.save(spath+"_cm2p.npy", cm2p)
    np.save(spath+"_cm4p.npy", cm4p)
    np.save(spath+"_cm2i.npy", cm2i)
    np.save(spath+"_cm4i.npy", cm4i)
else:
    cm2, cm4 = eval_dict[seval](subnum, train_data,models[0],models[1])
    np.save(spath+"_cm2.npy", cm2)
    np.save(spath+"_cm4.npy", cm4)