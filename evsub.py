import numpy as np
import argparse,pickle
import multi.modules.evaluatesub as es
from multi.modules.models import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=int, required=True)
parser.add_argument("--clf", type=str, required=True,
    choices=["lr", "xdawn", "eegnet", "eegconf", "atteca", "eegnex"],)
parser.add_argument("--eval", type=str, required=True,
    choices=["ori", "tt", "csub", "in", "cin", "ll"],)
args = parser.parse_args()

subnum = args.sub
sclf = args.clf
seval = args.eval

eval_dict = {
    "ori": es.evaluate_all_samples,
    "tt": es.evaluate_times,
    "csub": es.evaluate_cross_subject,
    "in": es.evaluate_in,
    "cin": es.evaluate_cross_subject_in,
    "ll": es.evaluate_all_samples_learning,
}

models = get_model(sclf)

with open("multi/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

spath = f"multi/acc/sub/{sclf}_{seval}_{subnum}"

if seval in ["in", "cin"]:
    acc_p, acc_i = eval_dict[seval](subnum, train_data,models[0],models[1])
    np.save(spath+"_p.npy", acc_p)
    np.save(spath+"_i.npy", acc_i)
else:
    acc_ = eval_dict[seval](subnum, train_data,models[0],models[1])
    np.save(spath, acc_)