import numpy as np

def mae(y, p):
    y, p = np.asarray(y,float), np.asarray(p,float)
    return float(np.mean(np.abs(y-p)))

def rmse(y, p):
    y, p = np.asarray(y,float), np.asarray(p,float)
    return float(np.sqrt(np.mean((y-p)**2)))

def smape(y, p):
    y, p = np.asarray(y,float), np.asarray(p,float)
    d=np.abs(y)+np.abs(p)
    return float(np.mean(2*np.abs(y-p)/np.where(d==0,1,d))*100)

def r2(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    den=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/den) if den else 0.0

def directional_accuracy(y,p,reference=None):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if reference is None:
        reference=np.r_[y[0],y[:-1]]
    reference=np.asarray(reference,float)
    return float(np.mean(np.sign(y-reference)==np.sign(p-reference))*100)

def all_metrics(y,p,reference=None):
    return {"MAE":mae(y,p),"RMSE":rmse(y,p),"sMAPE":smape(y,p),
            "R2":r2(y,p),"DirectionalAccuracy":directional_accuracy(y,p,reference)}
