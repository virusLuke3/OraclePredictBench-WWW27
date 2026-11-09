"""Vector figures and LaTeX tables generated only from executed run records."""
import json,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from .common import ROOT,PROCESSED

ORDER=['Prior','LR','RF','GBM','Ensemble','D0','A1','A2','D_match']
NAMES={'Prior':'Prior','LR':'Logistic','RF':'RF','GBM':'LightGBM','Ensemble':'AG ensemble','D0':'Direct LLM','A1':'Tool agent','A2':'Agent + verifier','D_match':'Matched evidence'}
COLORS={'Prior':'#9A9A9A','LR':'#597887','RF':'#3E91B0','GBM':'#156082','Ensemble':'#042433','D0':'#AD7CC7','A1':'#7030A0','A2':'#008A42','D_match':'#C04F15'}

def workflow():
    dst=ROOT/'Results/Diagrams_Source_Files';dst.mkdir(exist_ok=True)
    boxes=[('Frozen public\nevents',0),('As-of features\n+ mature labels',1),('Question-disjoint\ntime partitions',2),('Models / planner\n/ verifier',3),('Paired loss\nand latency',4)]
    fig=plt.figure(figsize=(7.2,1.2));ax=fig.add_axes([.015,.05,.97,.9]);ax.set_xlim(-.1,10.8);ax.set_ylim(0,1);ax.axis('off')
    for text,index in boxes:
        x=2.2*index
        ax.add_patch(FancyBboxPatch((x,.22),1.9,.56,boxstyle='round,pad=.02,rounding_size=.08',facecolor='#E8F0F3',edgecolor='#156082'))
        ax.text(x+.95,.5,text,ha='center',va='center',fontsize=8.5)
        if index<4:ax.annotate('',xy=(x+2.16,.5),xytext=(x+1.94,.5),arrowprops=dict(arrowstyle='->',color='#042433',shrinkA=0,shrinkB=0))
    for ext in ['pdf','svg']:fig.savefig(dst/f'workflow.{ext}',bbox_inches='tight')
    plt.close(fig)
    doc=ET.Element('mxfile',host='app.diagrams.net');dia=ET.SubElement(doc,'diagram',name='Workflow');model=ET.SubElement(dia,'mxGraphModel');root=ET.SubElement(model,'root')
    ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    for text,x in boxes:
        cell=ET.SubElement(root,'mxCell',id=str(x+2),value=text,style='rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0F3;strokeColor=#156082;fontSize=14;',vertex='1',parent='1')
        ET.SubElement(cell,'mxGeometry',x=str(40+210*x),y='100',width='180',height='70',attrib={'as':'geometry'})
        if x:
            edge=ET.SubElement(root,'mxCell',id=f'e{x}',edge='1',parent='1',source=str(x+1),target=str(x+2),style='endArrow=classic;')
            ET.SubElement(edge,'mxGeometry',relative='1',attrib={'as':'geometry'})
    ET.ElementTree(doc).write(dst/'workflow.drawio',encoding='utf-8',xml_declaration=True)

def main():
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42})
    results=pd.read_csv(ROOT/'Results/Tables/main_results.csv').set_index('method').loc[ORDER]
    manifest=pd.read_parquet(ROOT/'metadata/split_manifest.parquet');selection=json.loads((ROOT/'metadata/selected_model.json').read_text());summary=json.loads((ROOT/'Results/Source_Data/statistical_summary.json').read_text())
    predictions=pd.read_parquet(ROOT/'Results/Source_Data/predictions_scored.parquet')
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.65),layout='constrained');a,b,c,d=axs.ravel()
    for y,s,color in [(2,'train','#156082'),(1,'validation','#C04F15'),(0,'test','#008A42')]:
        m=manifest[manifest.split.eq(s)];a.plot([m.proposal_time.min(),m.proposal_time.max()],[y,y],linewidth=7,color=color,solid_capstyle='butt')
        a.text(.02,(y+.72)/3.2,f'{s}: N={len(m)}, G={m.group_id.nunique()}',transform=a.transAxes,fontsize=8)
    a.set_ylim(-.5,2.7);a.set_yticks([]);a.xaxis.set_major_locator(mdates.YearLocator());a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));a.set_xlabel('Proposal time (UTC)');a.set_title('A  Time and information boundary',loc='left')
    a.spines[['top','right','left']].set_visible(False)
    for i,m in enumerate(ORDER):
        row=results.loc[m];b.errorbar(row.brier,i,xerr=[[row.brier-row.ci_low],[row.ci_high-row.brier]],fmt='o',color=COLORS[m],markersize=4,capsize=2)
    b.set_yticks(range(len(ORDER)),[NAMES[m] for m in ORDER]);b.invert_yaxis();b.set_xlabel('Brier score (lower is better)');b.set_title(f'B  Performance, N={summary["N"]}, G={summary["G"]}',loc='left');b.grid(axis='x',alpha=.2)
    test=manifest[manifest.split.eq('test')].sort_values(['proposal_time','sample_id']);cases=test.iloc[[0,len(test)//2,len(test)-1]].sample_id.tolist()
    subset=predictions[predictions.sample_id.isin(cases)];subset.to_csv(ROOT/'Results/Source_Data/prespecified_cases.csv',index=False)
    mean=subset.groupby(['sample_id','method']).p_used.mean().unstack();case_methods=['Ensemble','D0','A1','A2','D_match']
    for k,m in enumerate(case_methods):c.plot(np.arange(3)+(k-2)*.1,mean.loc[cases,m],linestyle='none',marker=['s','o','^','D','v'][k],color=COLORS[m],label=m,markersize=4)
    labels=subset.groupby('sample_id').label.first().loc[cases];c.scatter(range(3),labels,marker='_',s=140,color='black',label='Outcome')
    c.set_xticks(range(3),['First','Middle','Last']);c.set_ylim(-.08,1.08);c.set_ylabel('P(protocol disagreement)');c.set_title('C  Fixed chronological examples',loc='left');c.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False)
    offsets={'Prior':(-26,5),'LR':(5,4),'RF':(-16,-12),'GBM':(-24,5),'Ensemble':(5,-4),'D0':(-22,-12),'A1':(-18,-14),'A2':(-20,10),'D_match':(5,12)}
    for i,m in enumerate(ORDER):
        row=results.loc[m];d.scatter(max(row.latency_p50_ms,1e-4)/1000,row.brier,s=26,color=COLORS[m],marker='D' if m in ['A1','A2','D_match'] else 'o');d.annotate(m,(max(row.latency_p50_ms,1e-4)/1000,row.brier),xytext=offsets[m],textcoords='offset points',fontsize=8)
    d.set_xscale('log');d.margins(x=.22,y=.2);d.set_xlabel('Median latency (s, log scale)');d.set_ylabel('Brier score');d.set_title('D  Performance and latency',loc='left');d.grid(alpha=.2)
    for ax in [b,c,d]:ax.spines[['top','right']].set_visible(False)
    for ext in ['pdf','svg']:fig.savefig(ROOT/f'Results/Figures/main.{ext}',bbox_inches='tight')
    plt.close(fig);workflow()
    rows=[]
    for m in ORDER:
        r=results.loc[m];name=NAMES[m].replace('_','\\_')
        rows.append(f'{name} & {r.brier:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] & {r.delta_vs_best:+.4f} [{r.delta_ci_low:+.4f}, {r.delta_ci_high:+.4f}] & {r.latency_p50_ms:.3g} & {r.mean_llm_calls:.2f} & {100*r.fallback_rate:.1f}\\% \\\\')
    table='\\begin{tabular}{lccrrr}\n\\toprule\nMethod & Brier [95\\% CI] & $\\Delta$ [95\\% CI] & Median ms & LLM calls & Fallback \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (ROOT/'paper/tables').mkdir(exist_ok=True);(ROOT/'paper/tables/main_results.tex').write_text(table)
    # Calibration plots use each method's pooled individual runs, not averaged-p ensembles.
    fig,ax=plt.subplots(figsize=(5,4));cal=[]
    for m in ['Ensemble','D0','A1','A2','D_match']:
        data=predictions[predictions.method.eq(m)].copy();data['bin']=np.minimum((data.p_used*5).astype(int),4)
        for i,g in data.groupby('bin'):cal.append({'method':m,'bin':i,'method_outputs':len(g),'requests':g.sample_id.nunique(),'mean_p':g.p_used.mean(),'observed_rate':g.label.mean()})
        mm=pd.DataFrame(cal).query('method==@m');ax.plot(mm.mean_p,mm.observed_rate,'o-',label=m,color=COLORS[m])
    ax.plot([0,1],[0,1],'k--',linewidth=.8);ax.set(xlabel='Mean predicted probability',ylabel='Observed fraction',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout()
    pd.DataFrame(cal).to_csv(ROOT/'Results/Source_Data/calibration.csv',index=False)
    for ext in ['pdf','svg']:fig.savefig(ROOT/f'Results/Figures/calibration.{ext}')
    plt.close(fig)
    print('Generated vector figures, editable workflow, source data and result table.')

if __name__=='__main__':main()
