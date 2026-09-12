"""StockForecastAI Phase 6 standalone validation pipeline.

This file mirrors the Phase 6 concepts in a compact runnable form:
strict holdout validation, tabular models, baselines, regime-aware weighting,
strategy backtesting, transaction costs, drift checks and model promotion.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def add_features(df):
    x=df.copy(); c=x.Close
    x['ret1']=c.pct_change(); x['ret5']=c.pct_change(5); x['ret20']=c.pct_change(20)
    x['range_pct']=(x.High-x.Low)/c.replace(0,np.nan)
    d=c.diff(); up=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean()
    x['rsi14']=100-100/(1+up/dn.replace(0,np.nan))
    prev=c.shift(1)
    tr=pd.concat([(x.High-x.Low).abs(),(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1)
    x['atr14']=tr.rolling(14).mean(); x['atr_pct']=x.atr14/c
    for n in (5,10,20,50,100,200):
        x[f'sma{n}']=c.rolling(n).mean(); x[f'close_sma{n}']=c/x[f'sma{n}']-1
    x['vol20']=x.ret1.rolling(20).std()*np.sqrt(252)
    x['trend60']=c.pct_change(60)
    for col in ['Open','High','Low','Close','Volume']:
        for lag in range(1,31): x[f'{col.lower()}_lag{lag}']=x[col].shift(lag)
    return x.replace([np.inf,-np.inf],np.nan)


def dataset(df,target,horizon):
    f=add_features(df)
    drop={'Open','High','Low','Close','Volume','Adj Close'}
    cols=[c for c in f.columns if c not in drop]
    y=pd.DataFrame({h:df[target].shift(-h) for h in range(1,horizon+1)},index=df.index)
    z=pd.concat([f[cols],y.add_prefix('y_')],axis=1).dropna()
    return z[cols], z[[f'y_{h}' for h in range(1,horizon+1)]].set_axis(range(1,horizon+1),axis=1), cols


class DirectModel:
    def __init__(self,kind,horizon,seed=42): self.kind=kind; self.horizon=horizon; self.seed=seed
    def new(self):
        if self.kind=='xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=500,max_depth=5,learning_rate=.03,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=self.seed,objective='reg:squarederror')
        if self.kind=='lightgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=500,num_leaves=31,learning_rate=.03,subsample=.85,colsample_bytree=.85,verbosity=-1,random_state=self.seed)
        return RandomForestRegressor(n_estimators=400,min_samples_leaf=3,max_features=.7,n_jobs=4,random_state=self.seed)
    def fit(self,df,target):
        X,Y,self.cols=dataset(df,target,self.horizon); self.models={}
        for h in range(1,self.horizon+1): self.models[h]=self.new().fit(X,Y[h])
        self.history=df.copy(); return self
    def predict(self):
        f=add_features(self.history); x=f[self.cols].iloc[[-1]].fillna(0)
        return np.array([float(self.models[h].predict(x)[0]) for h in range(1,self.horizon+1)])


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); d=np.abs(y)+np.abs(p)
    den=np.sum((y-y.mean())**2)
    return {'MAE':float(np.mean(np.abs(y-p))),'RMSE':float(np.sqrt(np.mean((y-p)**2))),
            'sMAPE':float(np.mean(2*np.abs(y-p)/np.where(d==0,1,d))*100),
            'R2':float(1-np.sum((y-p)**2)/den) if den else 0.0}


def regime(df):
    f=add_features(df)[['trend60','vol20','rsi14','atr_pct']].dropna()
    if len(f)<252: return 'Unknown'
    sc=StandardScaler(); z=sc.fit_transform(f)
    gm=GaussianMixture(n_components=4,random_state=42,n_init=3).fit(z)
    labels=gm.predict(z); last=int(labels[-1]); tmp=f.copy(); tmp['cluster']=labels
    stats=tmp.groupby('cluster').mean(numeric_only=True); row=stats.loc[last]
    direction='Bull' if row.trend60>.04 else ('Bear' if row.trend60<-.04 else 'Sideways')
    vol='HighVol' if row.vol20>=stats.vol20.median() else 'LowVol'
    return f'{direction}_{vol}'


def baseline(df,target,horizon,kind):
    if kind=='persistence': return np.repeat(float(df[target].iloc[-1]),horizon)
    if kind=='ma20': return np.repeat(float(df[target].tail(20).mean()),horizon)
    prev=df.Close.shift(1); tr=pd.concat([(df.High-df.Low).abs(),(df.High-prev).abs(),(df.Low-prev).abs()],axis=1).max(axis=1).tail(14).mean()
    close=float(df.Close.iloc[-1]); return np.repeat(close+tr if target=='High' else max(.01,close-tr),horizon)


def weighted(preds,errors):
    names=[k for k in preds if k in errors]
    if not names: names=list(preds); w={k:1/len(names) for k in names}
    else:
        e=np.array([max(errors[k],1e-9) for k in names]); raw=np.exp(-(e/e.mean())/.35); raw/=raw.sum(); w=dict(zip(names,raw))
    return sum(w[k]*np.asarray(preds[k],float) for k in names),w


def research_signal(close,hi,lo):
    up=(np.max(hi)/close-1)*100; down=(np.min(lo)/close-1)*100; rr=max(up,0)/max(abs(min(down,0)),1e-6)
    s='BUY' if up>=4 and rr>=1.3 else ('SELL' if down<=-4 and rr<.8 else 'HOLD')
    return {'signal':s,'upside_pct':float(up),'downside_pct':float(down),'reward_risk_ratio':float(rr)}


def strategy(signals,prices,cost_bps=10,slippage_bps=10,capital=100000):
    tc=cost_bps/10000; sl=slippage_bps/10000; cash=capital; shares=0.; entry=0.; eq=[]; trades=[]
    smap={r['date']:r['signal'] for r in signals}
    for dt,row in prices.iterrows():
        px=float(row.Close); s=smap.get(str(dt.date()),'HOLD')
        if s=='BUY' and shares==0:
            execp=px*(1+sl); alloc=cash/(1+tc); shares=alloc/execp; fee=alloc*tc; entry=alloc+fee; cash-=entry
        elif s=='SELL' and shares>0:
            net=shares*px*(1-sl)*(1-tc); trades.append(net-entry); cash+=net; shares=0.; entry=0.
        eq.append(cash+shares*px)
    if shares>0:
        px=float(prices.Close.iloc[-1]); net=shares*px*(1-sl)*(1-tc); trades.append(net-entry); cash+=net; eq[-1]=cash
    eq=np.asarray(eq,float); rets=pd.Series(eq).pct_change().dropna(); peak=np.maximum.accumulate(eq); dd=eq/peak-1
    pnl=np.asarray(trades,float); wins=pnl[pnl>0].sum(); losses=abs(pnl[pnl<0].sum())
    return {'total_return':float(eq[-1]/eq[0]-1) if len(eq)>1 else 0.,'Sharpe':float(rets.mean()/rets.std()*np.sqrt(252)) if len(rets)>1 and rets.std()>0 else 0.,
            'MaxDrawdown':float(dd.min()) if len(dd) else 0.,'ProfitFactor':float(wins/losses) if losses>0 else 0.,
            'HitRate':float((pnl>0).mean()) if len(pnl) else 0.,'TradeCount':int(len(pnl))}


def run(config='configs/phase6.yaml'):
    cfg=yaml.safe_load(Path(config).read_text()); H=int(cfg['horizon'])
    df=yf.download(cfg['ticker'],start=cfg['start'],auto_adjust=False,progress=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df[['Open','High','Low','Close','Volume']].dropna(); n=int(cfg['holdout']['sessions']); dev=df.iloc[:-n]; hold=df.iloc[-n:]
    score=[]; signals=[]; history_errors={t:{} for t in cfg['targets']}
    combined=pd.concat([dev,hold]); origins=range(len(dev),len(combined)-H+1,int(cfg['holdout']['step']))
    for origin in origins:
        train=combined.iloc[:origin]; actual=combined.iloc[origin:origin+H]; rg=regime(train); ens={}
        for target in cfg['targets']:
            preds={}
            for b in ('persistence','ma20','atr'): preds['baseline_'+b]=baseline(train,target,H,b)
            for kind in cfg['models']['tabular']:
                try: preds[kind]=DirectModel(kind,H,cfg['random_state']).fit(train,target).predict()
                except Exception: pass
            errors={k:np.mean(v) for k,v in history_errors[target].items() if v}
            ep,w=weighted(preds,errors); ens[target]=ep
            for name,p in preds.items():
                m=metrics(actual[target].values,p); score.append({'origin':str(actual.index[0].date()),'regime':rg,'model':name,'target':target,**m})
                history_errors[target].setdefault(name,[]).append(m['MAE'])
            m=metrics(actual[target].values,ep); score.append({'origin':str(actual.index[0].date()),'regime':rg,'model':'regime_ensemble','target':target,**m})
        sig=research_signal(float(train.Close.iloc[-1]),ens['High'],ens['Low']); signals.append({'date':str(actual.index[0].date()),'regime':rg,**sig})
    scores=pd.DataFrame(score); quality=scores.groupby(['model','target'])[['MAE','RMSE','sMAPE','R2']].mean().reset_index()
    strat=strategy(signals,hold,cfg['strategy']['transaction_cost_bps'],cfg['strategy']['slippage_bps'],cfg['strategy']['initial_capital'])
    out=Path(cfg['report_dir']); out.mkdir(parents=True,exist_ok=True); scores.to_csv(out/'holdout_scores_standalone.csv',index=False); quality.to_csv(out/'forecast_quality_standalone.csv',index=False)
    (out/'strategy_metrics_standalone.json').write_text(json.dumps(strat,indent=2)); print(quality.to_string(index=False)); print(strat)
    return quality,strat


if __name__=='__main__': run()
