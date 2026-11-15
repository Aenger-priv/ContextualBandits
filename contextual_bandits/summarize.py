"""Generate paired metrics from a locally produced confirmation bundle."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
from .evaluate import primary_contrasts

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=Path('runs/study'));a=p.parse_args()
    freeze=json.loads((a.study/'configs/freeze.json').read_text())
    rows=json.loads((a.study/'results/confirmation/summary.json').read_text())
    out=a.study/'analysis';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.groupby('method')[['expected_mean','realized_mean','regret']].mean().to_csv(out/'overall.csv')
    names,values=primary_contrasts(rows,freeze['selected']);n=len(values)
    half=t.ppf(1-.05/(2*len(names)),n-1)*values.std(axis=0,ddof=1)/np.sqrt(n)
    mean=values.mean(axis=0)
    result=pd.DataFrame({'contrast':names,'effect':mean,'lower':mean-half,'upper':mean+half,'worlds':n})
    result.to_csv(out/'paired-contrasts.csv',index=False)
    print(frame.groupby('method')[['expected_mean','regret']].mean().round(5).to_string())
    print(result.round(6).to_string(index=False));print('Saved:',out)
if __name__=='__main__':main()
