"""Regenerates docs/decoding_sweep.png from experiments/decoding_sweep/sweep_results.csv. Run from the repository root: python docs/make_sweep_figure.py"""
import matplotlib as mpl
S=['#2a78d6','#eb6834','#1baf7a','#eda100']
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; SURF='#ffffff'
mpl.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,
 'font.family':'DejaVu Sans','font.size':11,'axes.edgecolor':GRID,'axes.labelcolor':INK2,
 'xtick.color':INK2,'ytick.color':INK2,'axes.titlecolor':INK,'axes.titlesize':13,'axes.titleweight':'bold',
 'axes.titlelocation':'left','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,
 'grid.linewidth':0.8,'axes.axisbelow':True,'legend.frameon':False,'legend.labelcolor':INK2})

import pandas as pd, numpy as np, matplotlib.pyplot as plt
d=pd.read_csv('experiments/decoding_sweep/sweep_results.csv')
temps=sorted(d.temperature.unique())
fig,ax=plt.subplots(figsize=(9,4.6))
off={1:-0.15,2:0,3:0.15}
idx={t:i for i,t in enumerate(temps)}
lab={1:'Sampling / greedy (1 beam)',2:'2 beams',3:'3 beams'}
for i,b in enumerate([1,2,3]):
    g=d[d.num_beams==b]
    ax.scatter(g.temperature.map(idx)+off[b],100*g.overall_acc,s=26,color=S[i],alpha=0.45,edgecolor='white',linewidth=0.8,zorder=3)
    m=g.groupby('temperature').overall_acc.mean()*100
    ax.plot(np.array([idx[t] for t in m.index])+off[b],m.values,color=S[i],lw=2,marker='o',ms=6,mec='white',mew=1.5,label=lab[b],zorder=4)
ax.axhline(100*d.overall_acc.mean(),color=INK2,lw=1,ls=(0,(4,3)),zorder=2)
ax.text(6.45,100*d.overall_acc.mean(),f'mean of all 126\nruns: {100*d.overall_acc.mean():.1f} %',va='center',fontsize=9.5,color=INK2)
ax.set_xticks(range(len(temps)),[f'{t:g}' for t in temps]); ax.set_xlim(-0.45,7.6)
ax.set_xlabel('Temperature'); ax.set_ylabel('Progress (% of the 15 levels)')
ax.grid(axis='x',visible=False)
ax.legend(loc='upper left',ncol=3,fontsize=10,bbox_to_anchor=(0,1.0))
ax.set_ylim(10,42)
fig.suptitle('Decoding sweep: 126 configurations of Mistral-7B-Instruct',x=0.01,ha='left',fontsize=14,fontweight='bold',color=INK)
fig.text(0.01,0.875,'Dots: single runs, one per top-p value. Lines: mean per setting.',color=INK2,fontsize=10)
fig.tight_layout(rect=(0,0,1,0.88)); fig.savefig('docs/decoding_sweep.png',dpi=150)
