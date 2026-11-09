import numpy as np
import multi.modules.evaluatesub as es
from mne.decoding import XdawnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from braindecode.classifier import EEGClassifier
from braindecode.models import EEGNetv4,EEGConformer,AttentionBaseNet,EEGNeX
from sklearn.base import BaseEstimator, TransformerMixin, clone

class MyReshapeMean(BaseEstimator, TransformerMixin):
    def __init__(self, n_channels=30, n_feats=3, window=50):
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.window = window

    def fit(self, X, y=None):
        return self  # 不需要训练

    def transform(self, X):
        n_trials = len(X)
        # X 必须是 (trials, n_channels*n_feats*window) 或 (trials, n_channels, n_feats*window)
        X_reshaped = X.reshape(n_trials, self.n_channels, self.n_feats, self.window)
        X_out = X_reshaped.mean(axis=-1)  # 在 window 维求均值
        return X_out.reshape(n_trials, -1)

class ChannelZScore(BaseEstimator, TransformerMixin):
    """
    Standardize each channel across ALL trials and time points.
    Input X shape: (n_trials, n_chans, n_times)
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        # 转换成 numpy 数组
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        # 防止除以 0
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        # 使用 fit 阶段学到的全局参数进行转换
        return (X - self.mean_) / self.std_

clf_xdawn1 = make_pipeline(
    XdawnTransformer(n_components=3),
    MyReshapeMean(n_channels=6, n_feats=3, window=50),
    MinMaxScaler(),
    LogisticRegression(solver="lbfgs", max_iter=2000, C=0.1))
clf_xdawn2 = make_pipeline(
    XdawnTransformer(n_components=3),
    MyReshapeMean(n_channels=12, n_feats=3, window=50),
    MinMaxScaler(),
    LogisticRegression(solver="lbfgs", max_iter=2000, C=0.1))

clf_lr1 = make_pipeline(
    MyReshapeMean(n_channels=30, n_feats=3, window=50),
    MinMaxScaler(),
    LogisticRegression(solver="lbfgs", max_iter=2000, C=0.1))
clf_lr2 = make_pipeline(
    MyReshapeMean(n_channels=30, n_feats=3, window=50),
    MinMaxScaler(),
    LogisticRegression(solver="lbfgs", max_iter=2000, C=0.1))

model1 = EEGNetv4(n_outputs=2,n_chans=30,n_times=150,F1=8,D=2,F2=16,kernel_length=16)
clf_eegnet1 = make_pipeline(
    ChannelZScore(),
    EEGClassifier(module=model1,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))
model2 = EEGNetv4(n_outputs=4,n_chans=30,n_times=150,F1=8,D=2,F2=16,kernel_length=16)
clf_eegnet2 = make_pipeline(
    ChannelZScore(),
    EEGClassifier(module=model2,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))

model1 = EEGConformer(n_outputs=2,n_chans=30,n_times=150,n_filters_time=32,filter_time_length=15,
    pool_time_length=20,pool_time_stride=8,drop_prob=0.3,num_layers=2,num_heads=8,att_drop_prob=0.3)
clf_eegconf1 = make_pipeline(ChannelZScore(),
    EEGClassifier(module=model1,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))
model2 = EEGConformer(n_outputs=4,n_chans=30,n_times=150,n_filters_time=32,filter_time_length=15,
    pool_time_length=20,pool_time_stride=8,drop_prob=0.3,num_layers=2,num_heads=8,att_drop_prob=0.3)
clf_eegconf2 = make_pipeline(ChannelZScore(),
    EEGClassifier(module=model2,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))

model1 = AttentionBaseNet(n_outputs=2,n_chans=30,n_times=150,
    n_temporal_filters=32,temp_filter_length_inp=15,spatial_expansion=1,pool_length_inp=20,pool_stride_inp=8,drop_prob_inp=0.3,
    ch_dim=16,temp_filter_length=9,pool_length=4,pool_stride=4,drop_prob_attn=0.3,attention_mode="eca",kernel_size=7)
clf_atteca1 = make_pipeline(ChannelZScore(),
    EEGClassifier(module=model1,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))
model2 = AttentionBaseNet(n_outputs=4,n_chans=30,n_times=150,
    n_temporal_filters=32,temp_filter_length_inp=15,spatial_expansion=1,pool_length_inp=20,pool_stride_inp=8,drop_prob_inp=0.3,
    ch_dim=16,temp_filter_length=9,pool_length=4,pool_stride=4,drop_prob_attn=0.3,attention_mode="eca",kernel_size=7)
clf_atteca2 = make_pipeline(ChannelZScore(),
    EEGClassifier(module=model2,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))

model1 = EEGNeX(n_outputs=2,n_chans=30,n_times=150)
clf_eegnex1 = make_pipeline(ChannelZScore(),
    EEGClassifier(module=model1,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))
model2 = EEGNeX(n_outputs=4,n_chans=30,n_times=150)
clf_eegnex2 = make_pipeline(ChannelZScore(),
    EEGClassifier(module=model2,optimizer__lr=1e-3,batch_size=16,max_epochs=100,train_split=None,device="cuda"))

clf_dict = {
    "lr": (clf_lr1, clf_lr2),
    "xdawn": (clf_xdawn1, clf_xdawn2),
    "eegnet": (clf_eegnet1, clf_eegnet2),
    "eegconf": (clf_eegconf1, clf_eegconf2),    
    "atteca": (clf_atteca1, clf_atteca2),
    "eegnex": (clf_eegnex1, clf_eegnex2),
}

def get_model(name):
    if name not in clf_dict:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available models: {list(clf_dict.keys())}"
        )
    return tuple(clone(clf) for clf in clf_dict[name])

eval_dict = {
    "ori": es.evaluate_all_samples,
    "tt": es.evaluate_times,
    "csub": es.evaluate_cross_subject,
    "in": es.evaluate_in,
    "cin": es.evaluate_cross_subject_in,
    "ll": es.evaluate_all_samples_learning,
}