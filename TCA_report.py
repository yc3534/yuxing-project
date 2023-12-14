#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:10:54 2023

12/14/2023 modification:
- add vwap_5min, vwap_15min, vwap_30min
- add trading time anaylsis
- output trading volume, aum, commission fee, tax 

@author: CHEN, Yuxing
"""
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from time import time
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_axes_aligner
from pathlib import Path
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")
opj = os.path.join
from aqumon_sdk.common.constant import OMSEnvironment
from aqumon_sdk.trade.trade_context import TradeBase

ROOT = Path(__file__).resolve().parents[0]
DATAPATH = opj(ROOT, 'data')
Path(DATAPATH).mkdir(parents=True, exist_ok=True)

trade = TradeBase(OMSEnvironment.PROD)
query = trade.account.query_history_execution

FIELD_DICT= {'open_BS': ['open_slippage', 'open_slippage_%'],
            'open_B': ['buy_open_slippage', 'buy_open_slippage_%'],
            'open_S': ['sell_open_slippage', 'sell_open_slippage_%'], 
            'vwap_BS': ['vwap_slippage', 'vwap_slippage_%'],
            'vwap_B': ['buy_vwap_slippage', 'buy_vwap_slippage_%'],
            'vwap_S': ['sell_vwap_slippage', 'sell_vwap_slippage_%']}

priceDir, fieldL = list(FIELD_DICT.keys()), list(FIELD_DICT.values())
fields = sum(fieldL, [])
nrow, ncol = 2,3

def get_tca_data(account, start, end):
    print(f'{account}: get tca data from {start} to {end}')
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    start_ts = int(round(start.timestamp() *1000))
    end_ts = int(round(end.timestamp() *1000))
    req = query(start_ts, end_ts, portfolio_accounts=[account])
    dates = sorted(req[account].keys())
    if len(dates) == 0:
        print(f'{account}: no transactions between {start} and {end}!')
        return None
    
    dfs = []
    for dt in dates:
        dt_str = pd.to_datetime(dt).strftime('%Y%m%d')
        url_txt = f'https://algo-internal.aqumon.com/trading/aim-gds/report-engine/tradingAnalysis/analyseAccount?accountMnemonic={account}&date={dt_str}&aggregate=true'
        r = requests.get(url_txt).json()
        if r['data'] == []:
            print(f'{dt}: no data returns!')
        else:
            try:
                data = r['data'][0]['tradingTimeslotSlippages'][0]
                df_dt = pd.DataFrame(data.items()).set_index(0).loc[fields].T
                df_dt.index = [pd.to_datetime(dt)]
                df_dt.columns.name=None
                dfs += [df_dt]
            except Exception as e:
                        print(f'{dt}:{e}')
    return pd.concat(dfs, axis=0)

def update_tca_data(accounts, start, end):
    for account in accounts:
        if Path(opj(DATAPATH, f'{account}.csv')).is_file():
            df = pd.read_csv(opj(DATAPATH, f'{account}.csv'), index_col=0, parse_dates=True)
            st, ed = start, df.index[0].strftime('%Y%m%d')
            if st < ed:
                df_period = get_tca_data(account, st, ed)
                if df_period is not None:
                    df = df.combine_first(df_period)
            st, ed = df.index[-1].strftime('%Y%m%d'), end
            if st < ed:
             df_period = get_tca_data(account, st, ed)
             if df_period is not None:
                 df = df.combine_first(df_period)
        else:
            df = get_tca_data(account, start, end)
        df.to_csv(opj(DATAPATH, f'{account}.csv'))
    return 'Done!'
            

def update_tca_data_mp(account_list, start, end):
    t0 = time()
    
    nprocs = min(4, len(account_list))
    split = np.array_split(np.array(account_list), nprocs)
    
    procs = []
    for i in range(nprocs):
        p = mp.Process(
            target=update_tca_data,
            args=(list(split[i]),
                  start,
                  end))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
    t1 = time()
    print(f'Complete TCA data updating，耗时{(t1-t0)/60:.2f}分钟，平均{(t1-t0)/60/len(account_list):.2f}分钟。')
    return 'Done!'
      
def plot_tca(account, start, end):
    df = pd.read_csv(opj(DATAPATH, f'{account}.csv'), index_col=0, parse_dates=True)[start:end]
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor('cornflowerblue')
    
    for i in range(nrow*ncol):
        field = priceDir[i]
        cols = fieldL[i]
        df_ = df[cols]
        idx, slp_l, slp_pct_l = df_.index.tolist(), df_.iloc[:,0].values, df_.iloc[:,1].values
        transDir, price = field[field.find('_')+1:], field[:field.find('_')]
        print(field, cols, transDir, price)
        
        if transDir == 'BS':
            title = f'Buy & Sell Transactions (marketPrice={price})'
        elif transDir == 'B':
            title = f'Buy Transactions (marketPrice={price})'
        elif transDir == 'S':
            title = f'Sell Transactions (marketPrice={price})'
        
        
        ax = fig.add_subplot(nrow, ncol, i+1)
        axR = ax.twinx()
        
        lL = ax.plot(idx, slp_l, '-b', label = f"slippage:{np.mean(slp_l):.2f}")
        ax.set(xlabel='date', ylabel='slippage', title=title)
        # plt.xticks(idx, rotation=45)
        ax.set_xticklabels(idx, rotation=45, ha='right')
    
        lR = axR.plot(idx, slp_pct_l, '-r', label = f"slippage %: {np.mean(slp_pct_l)*100:.2f} %")
        axR.axhline(y=0, color='grey', linestyle='--')
        axR.set(ylabel='slippage %')
        axR.set_yticklabels(['{:,.2%}'.format(x) for x in axR.get_yticks()])
        
        # Align y = 0 of ax1 and ax2 with the center of figure.
        mpl_axes_aligner.align.yaxes(ax, 0, axR, 0, 0.5)
        
        # legend
        lns = lL + lR
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        
        # text
        gapL = (ax.get_yticks()[1]-ax.get_yticks()[0])/2
        gapR = (axR.get_yticks()[1]-axR.get_yticks()[0])/2
        for k,v in zip(idx, slp_l):
            ax.text(k, v+0.5*gapL, round(v,0), ha='center', va='center',color = 'blue')
            
        for k,v in zip(idx, slp_pct_l):
            axR.text(k, v-0.5*gapR, f'{round(v*100, 2)}%', ha='center', va='center',color = 'red')

    plt.suptitle(f'{account}', size = 16)
    fig.tight_layout()
    return fig

def kf_stat(account_list, kf, start, end):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor('cornflowerblue')
    
    for i in range(nrow*ncol):
        idxs, kf_l, no_kf_l = [], [], []
        field = priceDir[i]
        cols = fieldL[i]
        ax = fig.add_subplot(nrow, ncol, i+1)
        for account in account_list:
            df = pd.read_csv(opj(DATAPATH, f'{account}.csv'), 
                             index_col=0, parse_dates=True)[start:end]
            df_ = df[cols]
            idx, slp_pct_l = df_.index.tolist(), df_.iloc[:,1].values
            transDir, price = field[field.find('_')+1:], field[:field.find('_')]
            print(field, cols, transDir, price)
            
            if transDir == 'BS':
                title = f'Buy & Sell Transactions (marketPrice={price})'
            elif transDir == 'B':
                title = f'Buy Transactions (marketPrice={price})'
            elif transDir == 'S':
                title = f'Sell Transactions (marketPrice={price})'
            
            if account in kf:
                ax.scatter(idx, slp_pct_l, color='red', alpha=0.4)
                kf_l += [df_.iloc[:, 1]]
            else:
                ax.scatter(idx, slp_pct_l, color='blue', alpha=0.4)
                no_kf_l += [df_.iloc[:, 1]]
            idxs += idx
        kf_avg = pd.concat(kf_l, axis=1).mean(axis=1)
        no_kf_avg = pd.concat(no_kf_l, axis=1).mean(axis=1)
        kf_avg.name = f'w/ kaifang:{kf_avg.mean()*100:.2f}%'
        no_kf_avg.name = f'w/o kaifang:{no_kf_avg.mean()*100:.2f}%'
        ax.plot(kf_avg.index, kf_avg.values, '-r',label = f'w/ kaifang:{kf_avg.mean()*100:.2f}%')
        ax.plot(no_kf_avg.index, no_kf_avg.values, '-b',label = f'w/o kaifang:{no_kf_avg.mean()*100:.2f}%')
        ax.legend()
        
        idxs = np.sort(list(set(idxs)))
        ax.set(xlabel='date', ylabel='slippage %', title=title)
        ax.set_xticklabels(idxs, rotation=45, ha='right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax.axhline(y=0, color='grey', linestyle='--')

    plt.suptitle('w/ vs w/o kafang', size = 16)
    fig.tight_layout()
    return fig

def tca_report(account_list, kf, start, end):
    update_tca_data_mp(account_list, start, end)
    plts = []
    for acc in account_list:
        plts.append(plot_tca(acc, start, end))
    plts.append(kf_stat(account_list, kf, start, end))
    
    text = '''
                <html>
                    <body>
                    </body>
                </html>
                '''
    for fig in plts:
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png', edgecolor=fig.get_edgecolor())
        tmpfile.seek(0)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        
        bf = text[:text.find('</body>\n')-16]
        af = text[text.find('</body>\n')-16:]
        add = f"                    <img src=\'data:image/png;base64,{encoded}'>\n"
        text = bf + add + af
    
        
    with open(opj(ROOT, f'tca_report_{start}_{end}.html'), 'w') as f:
        f.write(text)   

#%%          
if __name__ == '__main__':
    start, end = '20230713', '20231213'
    account_list = ['td.youxuanNo1-zx-normal-mf', 
                    'td.youxuanNo1-xiangcai-mf', 
                    'td.zz500-guojun-mf', 
                    'td.zz500-pingan-mf', 
                    'td.hs300-huatai-normal-mf', 
                    'td.johnsonlong-cicc-mf', 
                    'td.sumlong-cicc-mf', 
                    "td.jingwaiSp1-ms-mf"]
    
    kf = ['td.youxuanNo1-zx-normal-mf', 
        'td.youxuanNo1-xiangcai-mf', 
        'td.zz500-guojun-mf', 
        'td.zz500-pingan-mf', 
        'td.hs300-huatai-normal-mf']

    tca_report(account_list, kf, start, end)
    
    

    
        