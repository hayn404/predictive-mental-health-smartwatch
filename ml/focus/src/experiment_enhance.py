"""
Experiment harness to enhance the Samsung focus model under honest LOSO-CV.
Tests: enhanced RR extraction (upsample + tighter band) vs original, window length,
robust per-user normalization, and per-session temporal aggregation.

Run: python3 ml/focus/src/experiment_enhance.py
"""
import os, sys, math, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy import signal as ss, interpolate
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(BASE, 'stress'))
from src.features import compute_time_domain, compute_nonlinear  # noqa

ROOT = os.path.join(BASE, 'focus', 'data', 'cogwear',
                    'cogwear-can-we-detect-cognitive-effort-with-consumer-grade-wearables-1.0.0')
FEATS = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','hrStd','hrRange','cvRR',
         'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']


def rr_original(bvp, fs):
    nyq = fs/2
    b,a = ss.butter(3,[0.5/nyq, min(8.0/nyq,0.99)],btype='band')
    f = ss.filtfilt(b,a,bvp)
    pk,_ = ss.find_peaks(f, distance=int(0.3*fs), height=np.percentile(f,60))
    if len(pk)<3: return np.array([])
    rr = np.diff(pk)/fs*1000
    rr = rr[(rr>=300)&(rr<=2000)]
    if len(rr)>5:
        m=np.median(rr); rr=rr[np.abs(rr-m)/m<0.3]
    return rr


def rr_enhanced(bvp, fs, target=128):
    """Upsample filtered PPG to 128 Hz (cubic) before peak detection -> finer RR timing."""
    nyq = fs/2
    b,a = ss.butter(3,[0.5/nyq, min(4.0/nyq,0.99)],btype='band')  # tighter HR band
    f = ss.filtfilt(b,a,bvp)
    n=len(f); t=np.arange(n)/fs
    if t[-1] <= 0: return np.array([])
    t2=np.arange(0,t[-1],1/target)
    up=interpolate.interp1d(t,f,kind='cubic',fill_value='extrapolate')(t2)
    pk,_=ss.find_peaks(up, distance=int(0.33*target), height=np.percentile(up,55))
    if len(pk)<3: return np.array([])
    rr=np.diff(pk)/target*1000
    rr=rr[(rr>=300)&(rr<=2000)]
    if len(rr)>5:
        m=np.median(rr); rr=rr[np.abs(rr-m)/m<0.3]
    return rr


def feats_from_rr(rr):
    if len(rr)<20: return None
    d={}; d.update(compute_time_domain(rr)); d.update(compute_nonlinear(rr))
    return d


def extract(method, win, step):
    rows=[]
    def cond_dirs():
        pilot=os.path.join(ROOT,'pilot')
        for sid in sorted(os.listdir(pilot)):
            for c,l in [('baseline',0),('cognitive_load',1)]:
                yield os.path.join(pilot,sid,c),sid,l
        surv=os.path.join(ROOT,'survey_gamification')
        for sid in sorted(os.listdir(surv)):
            for s in ['pre','post']:
                for c,l in [('baseline',0),('cognitive_load',1)]:
                    yield os.path.join(surv,sid,s,c),sid,l
    rrfn = rr_enhanced if method=='enhanced' else rr_original
    for folder,sid,label in cond_dirs():
        p=os.path.join(folder,'samsung_bvp.csv')
        if not os.path.exists(p): continue
        df=pd.read_csv(p)
        if 'PPG GREEN' not in df or len(df)<200: continue
        bvp=df['PPG GREEN'].to_numpy(float); t=df['time'].to_numpy(float)
        fs=len(t)/(t[-1]-t[0]) if t[-1]>t[0] else 0
        if fs<5: continue
        seg_id=0; start=t[0]
        while start+win<=t[-1]:
            m=(t>=start)&(t<start+win); s=bvp[m]
            if len(s)>=0.6*win*fs:
                rr=rrfn(s,fs); fe=feats_from_rr(rr)
                if fe: fe.update({'subject_id':f'COG_{sid}','label':label,'seg':f'{sid}_{folder[-20:]}_{seg_id}'}); rows.append(fe)
            seg_id+=1; start+=step
    return pd.DataFrame(rows)


def pu_norm(df, robust=False):
    o=df.copy()
    for f in FEATS:
        g=o.groupby('subject_id')[f]
        if robust:
            med=g.transform('median'); iqr=(g.transform(lambda x:x.quantile(.75))-g.transform(lambda x:x.quantile(.25))).replace(0,1).fillna(1)
            o[f]=(o[f]-med)/iqr
        else:
            o[f]=(o[f]-g.transform('mean'))/g.transform('std').replace(0,1).fillna(1)
    return o.fillna(0)


PARAMS=dict(objective='binary:logistic',eval_metric='logloss',n_estimators=400,
            learning_rate=0.05,max_depth=2,subsample=0.8,colsample_bytree=0.8,
            reg_lambda=2.0,reg_alpha=0.5,gamma=0.2,random_state=42,n_jobs=-1)


def loso(df, aggregate=False):
    df=df.replace([np.inf,-np.inf],np.nan).fillna(0)
    X=df[FEATS].to_numpy(float); y=df['label'].to_numpy(int); g=df['subject_id'].to_numpy()
    logo=LeaveOneGroupOut(); yt,yp,ypr=[],[],[]
    for tr,te in logo.split(X,y,g):
        pos=max(1,(y[tr]==1).sum()); neg=max(1,(y[tr]==0).sum())
        m=xgb.XGBClassifier(**PARAMS,scale_pos_weight=neg/pos); m.fit(X[tr],y[tr])
        p=m.predict_proba(X[te])[:,1]
        if aggregate:
            # pool windows per (subject, condition-label) -> one prediction per state
            sub=df.iloc[te]; tmp=pd.DataFrame({'k':sub['subject_id']+'_'+sub['label'].astype(str),'p':p,'y':sub['label'].to_numpy()})
            agg=tmp.groupby('k').agg(p=('p','mean'),y=('y','first'))
            yt.extend(agg['y']); ypr.extend(agg['p']); yp.extend((agg['p']>=0.5).astype(int))
        else:
            yt.extend(y[te]); ypr.extend(p); yp.extend((p>=0.5).astype(int))
    yt,yp,ypr=map(np.array,(yt,yp,ypr))
    return accuracy_score(yt,yp),f1_score(yt,yp),matthews_corrcoef(yt,yp),roc_auc_score(yt,ypr)


if __name__=='__main__':
    print(f"{'config':42s} acc    F1     MCC    AUC")
    print("-"*72)
    configs=[
        ('orig  win60 step10 zscore', 'original',60,10,False,False),
        ('enh   win60 step10 zscore', 'enhanced',60,10,False,False),
        ('enh   win90 step15 zscore', 'enhanced',90,15,False,False),
        ('enh   win120 step20 zscore','enhanced',120,20,False,False),
        ('enh   win90 step15 robust', 'enhanced',90,15,True,False),
    ]
    cache={}
    for name,meth,win,step,robust,_ in configs:
        key=(meth,win,step)
        if key not in cache: cache[key]=extract(meth,win,step)
        df=cache[key]
        dn=pu_norm(df,robust)
        a,f,mc,au=loso(dn,aggregate=False)
        print(f"{name:42s} {a:.3f}  {f:.3f}  {mc:.3f}  {au:.3f}  (n={len(df)})")
    # best window + aggregation effect
    print("\n-- per-session temporal aggregation (deployment-style) --")
    for win,step in [(90,15),(120,20)]:
        df=cache[('enhanced',win,step)]; dn=pu_norm(df,False)
        a,f,mc,au=loso(dn,aggregate=True)
        print(f"{'enh win'+str(win)+' AGGREGATED':42s} {a:.3f}  {f:.3f}  {mc:.3f}  {au:.3f}")
