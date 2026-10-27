"""GPU observed-state delta head; masked rows keep the existing baseline."""
from pathlib import Path
import gc, json, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import gpu_nightly_runner as core

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, SUB = core.DATA, core.RESULTS, core.SUBMISSIONS
LOG = RESULTS / "delta_observed.jsonl"
SEEDS = (42, 77, 123, 2027, 9001)

def log(x):
    row={"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),**x}
    with LOG.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)

def score(y,p): return float(np.sqrt(np.mean((y-p)**2)))

def fold_data(train, fold):
    core.FIT_MASK_RATE=.9
    maxd=train.time.max()-pd.DateOffset(months=12*fold)
    cutoff=maxd-pd.DateOffset(months=12)+pd.DateOffset(days=1)
    valid0=train.loc[(train.time>=cutoff)&(train.time<=maxd)].copy()
    fit0=train.loc[train.time<cutoff].tail(5_000_000).copy()
    fit,valid,masked=core.apply_schedule(fit0,valid0,fold)
    combined=pd.concat([fit.drop(columns=['target']),valid.drop(columns=['target'])],ignore_index=True,sort=False)
    feat=core.build_features(combined).replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
    return feat.iloc[:len(fit)],fit,feat.iloc[len(fit):],valid,masked

def main():
    train=pd.read_csv(DATA/'Train.csv')
    train.time=pd.to_datetime(train.time)
    for seed in SEEDS:
        fold_scores=[]
        for fold in (0,1):
            xf,fit,xv,valid,masked=fold_data(train,fold)
            obs_fit=fit['TWS_t'].notna().to_numpy(); obs_valid=~masked
            ydelta=(fit.loc[obs_fit,'target']-fit.loc[obs_fit,'TWS_t']).to_numpy(np.float32)
            model=CatBoostRegressor(iterations=1000,depth=8,learning_rate=.04,l2_leaf_reg=10,loss_function='RMSE',task_type='GPU',devices='0',random_seed=seed,verbose=False,allow_writing_files=False)
            model.fit(xf.loc[obs_fit],ydelta,verbose=False)
            base=np.load(RESULTS/f'fold{fold}'/'direct_base.npy')
            route=base.copy(); route[obs_valid]=valid.loc[obs_valid,'TWS_t'].to_numpy()+model.predict(xv.loc[obs_valid])
            s=score(valid.target.to_numpy(float)[obs_valid],route[obs_valid]); fold_scores.append(s)
            log({'stage':'fold','seed':seed,'fold':fold,'observed_rmse':s,'n_fit':int(obs_fit.sum()),'n_valid':int(obs_valid.sum())})
            del model,xf,xv; gc.collect()
        log({'stage':'pair','seed':seed,'mean_observed_rmse':float(np.mean(fold_scores))})
    log({'stage':'complete'})
    while True: time.sleep(60)

if __name__=='__main__': main()
