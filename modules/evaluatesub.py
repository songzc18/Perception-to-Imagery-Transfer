import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.base import clone

func_map = {
    "decision_function": lambda clf, X: clf.decision_function(X),
    "predict_proba": lambda clf, X: clf.predict_proba(X)
}

def split_fold(X_, y_, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    out_data = []
    for train_idx, test_idx in skf.split(X_, y_):
        X_train = X_[train_idx,:,:150]
        y_train = y_[train_idx]
        X_test_no, y_test_no = [], []
        for jj in range(25):
            X_testno = X_[test_idx,:,750+jj*25:900+jj*25]
            y_testno = y_[test_idx]
            X_test_no.append(X_testno)
            y_test_no.append(y_testno)
        X_test_no = np.concatenate(X_test_no, axis=0)
        y_test_no = np.concatenate(y_test_no, axis=0)

        X_test_ti, y_test_ti = [], []
        for jj in range(16):
            X_testti,y_testti = [], []
            for tt in range(10):
                X_testt = X_[test_idx,:,750+jj*25+tt*25:900+jj*25+tt*25]
                y_testt = y_[test_idx]
                X_testti.append(X_testt)
                y_testti.append(y_testt)
            X_test_ti.append(X_testti)
            y_test_ti.append(y_testti)
        X_test_ti = np.concatenate(X_test_ti, axis=1)
        y_test_ti = np.concatenate(y_test_ti, axis=1)[0]
        fold_data = [X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti]
        out_data.append(fold_data)
    return out_data

def train_test(clf, X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti):
    clf.fit(X_train, y_train)
    score_no = clf.score(X_test_no, y_test_no)
    sall = []
    for tt in range(10):
        score_func1 = "decision_function" if hasattr(clf, "decision_function") else "predict_proba"
        s1 = func_map[score_func1](clf, X_test_ti[tt])
        sall.append(s1)
    score_m = np.mean(np.array(sall), axis=0) 
    y_pred_r = (score_m >= 0).astype(int) if score_m.ndim == 1 else np.argmax(score_m, axis=1)
    score_ti = accuracy_score(y_test_ti, y_pred_r)
    return score_no, score_ti

def train_test_flod(clf, split_folds):
    scores = []
    for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in split_folds:
        clf_fold = clone(clf)
        score_no, score_ti = train_test(clf_fold, 
                X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti)
        scores.append([score_no, score_ti])
    return np.array(scores)

def evaluate_all_samples(ss, data, clf2, clf4, n_splits=10):
    X2, y2, X4, y4 = data[ss]
    split_folds2 = split_fold(X2, y2, n_splits=n_splits)
    scores2 = train_test_flod(clf2, split_folds2)
    split_folds4 = split_fold(X4, y4, n_splits=n_splits)
    scores4 = train_test_flod(clf4, split_folds4)
    return np.array([scores2, scores4])

def evaluate_times(ss, data, clf2, clf4, n_splits=10):
    X2, y2, X4, y4 = data[ss]
    skf2 = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    acc_flod2 = []
    for train_idx, test_idx in skf2.split(X2, y2):
        clf2_clone = clone(clf2)
        X_train2 = X2[train_idx,:,:150]
        y_train2 = y2[train_idx]
        clf2_clone.fit(X_train2, y_train2)
        acc_tt = []
        for jj in range(35):
            X_test2 = X2[test_idx,:,500+jj*25:650+jj*25]
            y_test2 = y2[test_idx]
            score2 = clf2_clone.score(X_test2, y_test2)
            acc_tt.append(score2)
        acc_flod2.append(acc_tt)
    skf4 = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    acc_flod4 = []
    for train_idx, test_idx in skf4.split(X4, y4):
        clf4_clone = clone(clf4)
        X_train4 = X4[train_idx,:,:150]
        y_train4 = y4[train_idx]
        clf4_clone.fit(X_train4, y_train4)
        acc_tt = []
        for jj in range(35):
            X_test4 = X4[test_idx,:,500+jj*25:650+jj*25]
            y_test4 = y4[test_idx]
            score4 = clf4_clone.score(X_test4, y_test4)
            acc_tt.append(score4)
        acc_flod4.append(acc_tt)
    return np.array([acc_flod2, acc_flod4])

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
    acc2 = train_test(clf2_clone, X_train2, y_train2, X_test_no2, y_test_no2, X_test_ti2, y_test_ti2)
    acc4 = train_test(clf4_clone, X_train4, y_train4, X_test_no4, y_test_no4, X_test_ti4, y_test_ti4)
    return np.array([acc2, acc4])

def split_foldin(X_, y_, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    out_data_p, out_data_i = [],[]
    for train_idx, test_idx in skf.split(X_, y_):
        X_train_p = X_[train_idx,:,:150]
        X_test_p = X_[test_idx,:,:150]
        y_train_p = y_[train_idx]
        y_test_p = y_[test_idx]
        X_train_i, y_train_i, X_test_no, y_test_no = [], [], [], []
        for jj in range(25):
            X_traini = X_[train_idx,:,750+jj*25:900+jj*25]
            y_traini = y_[train_idx]
            X_testno = X_[test_idx,:,750+jj*25:900+jj*25]
            y_testno = y_[test_idx]
            X_train_i.append(X_traini)
            y_train_i.append(y_traini)
            X_test_no.append(X_testno)
            y_test_no.append(y_testno)
        #X_train_i = np.concatenate(X_train_i, axis=0)
        #y_train_i = np.concatenate(y_train_i, axis=0)
        X_test_no = np.concatenate(X_test_no, axis=0)
        y_test_no = np.concatenate(y_test_no, axis=0)

        X_test_ti, y_test_ti = [], []
        for jj in range(16):
            X_testti,y_testti = [], []
            for tt in range(10):
                X_testt = X_[test_idx,:,750+jj*25+tt*25:900+jj*25+tt*25]
                y_testt = y_[test_idx]
                X_testti.append(X_testt)
                y_testti.append(y_testt)
            X_test_ti.append(X_testti)
            y_test_ti.append(y_testti)
        X_test_ti = np.concatenate(X_test_ti, axis=1)
        y_test_ti = np.concatenate(y_test_ti, axis=1)[0]
        fold_data_p = [X_train_p, y_train_p, X_test_p, y_test_p]
        fold_data_i = [X_train_i, y_train_i, X_test_no, y_test_no, X_test_ti, y_test_ti]
        out_data_p.append(fold_data_p)
        out_data_i.append(fold_data_i)
    return out_data_p, out_data_i

def evaluate_in(ss, data, clf2, clf4, n_splits=10):
        X2, y2, X4, y4 = data[ss]
        dp2, di2 = split_foldin(X2, y2, n_splits=n_splits)
        scores2p = []
        for X_train_p, y_train_p, X_test_p, y_test_p in dp2:
            clf_fold2 = clone(clf2)
            clf_fold2.fit(X_train_p, y_train_p)
            scores2p.append(clf_fold2.score(X_test_p, y_test_p))
        scores2i = []
        for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in di2:
            clf_fold2 = clone(clf2)
            scores2i_flod = []
            for jj in range(25):
                score_no, score_ti = train_test(clf_fold2, 
                        X_train[jj], y_train[jj], X_test_no, y_test_no, X_test_ti, y_test_ti)
                scores2i_flod.append([score_no, score_ti])
            scores2i.append(scores2i_flod)
        # 4 class
        dp4, di4 = split_foldin(X4, y4, n_splits=n_splits)
        scores4p = []
        for X_train_p, y_train_p, X_test_p, y_test_p in dp4:
            clf_fold4 = clone(clf4)
            clf_fold4.fit(X_train_p, y_train_p)
            scores4p.append(clf_fold4.score(X_test_p, y_test_p))
        scores4i = []
        for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in di4:
            clf_fold4 = clone(clf4)
            scores4i_flod = []
            for jj in range(25):
                score_no, score_ti = train_test(clf_fold4, 
                        X_train[jj], y_train[jj], X_test_no, y_test_no, X_test_ti, y_test_ti)
                scores4i_flod.append([score_no, score_ti])
            scores4i.append(scores4i_flod)
        return np.array([scores2p, scores4p]), np.array([scores2i, scores4i])

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
    scores2p = clf2_clone.score(X_test2_p, y_test2)
    scores4p = clf4_clone.score(X_test4_p, y_test4)
    score_no_2, score_ti_2,score_no_4, score_ti_4 = [],[],[],[]
    for jj in range(25):
        clf2_clone = clone(clf2)
        clf4_clone = clone(clf4)
        score_no2, score_ti2 = train_test(clf2_clone, 
                            X_train2_i[jj], y_train2_i, X_test2_i_no, 
                            y_test2_i_no, X_test2_i_ti, y_test2_i_ti)
        score_no4, score_ti4 = train_test(clf4_clone,
                            X_train4_i[jj], y_train4_i, X_test4_i_no, 
                            y_test4_i_no, X_test4_i_ti, y_test4_i_ti)
        score_no_2.append(score_no2)
        score_ti_2.append(score_ti2)
        score_no_4.append(score_no4)
        score_ti_4.append(score_ti4)
    return np.array([scores2p, scores4p]), np.array([[score_no_2, score_ti_2],[score_no_4, score_ti_4]])

def train_test_flod_learning(clf, split_folds, fractions=np.arange(0.2, 1., 0.1), repeats=20):
    results = []
    for X_train, y_train, X_test_no, y_test_no, X_test_ti, y_test_ti in split_folds:
        flod_ = []
        for frac in fractions:
            fold_scores = []
            sss = StratifiedShuffleSplit(n_splits=repeats,train_size=frac,random_state=42)
            for sub_idx, _ in sss.split(X_train, y_train):
                clf_fold = clone(clf)
                score_no, score_ti = train_test(clf_fold,X_train[sub_idx],y_train[sub_idx],
                                                X_test_no,y_test_no,X_test_ti,y_test_ti)
                fold_scores.append([score_no, score_ti])
            flod_.append(fold_scores)
        results.append(flod_)
    return np.array(results)

def evaluate_all_samples_learning(ss, data, clf2, clf4, n_splits=10):
    X2, y2, X4, y4 = data[ss]
    split_folds2 = split_fold(X2, y2, n_splits=n_splits)
    scores2 = train_test_flod_learning(clf2, split_folds2)
    split_folds4 = split_fold(X4, y4, n_splits=n_splits)
    scores4 = train_test_flod_learning(clf4, split_folds4)
    return np.array([scores2, scores4])