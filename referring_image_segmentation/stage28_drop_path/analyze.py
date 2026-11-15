from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
 r=ROOT/'results';assert (r/'DONE.json').exists()
 summary=json.loads((r/'summary.json').read_text())
 with (r/'per_image.csv').open() as f:rows=list(csv.DictReader(f))
 by={n:{x['id']:x for x in rows if x['model']==n and x['criterion']=='best_dice'} for n in ['control','drop_path']}
 ids=list(by['control']);assert ids==list(by['drop_path']) and len(ids)==762
 areas=np.array([int(by['control'][sid]['tp'])+int(by['control'][sid]['fn']) for sid in ids])
 cuts=np.quantile(areas,[.25,.5,.75]);paired=[]
 for sid,area in zip(ids,areas):
  a,b=by['control'][sid],by['drop_path'][sid]
  assert int(b['tp'])+int(b['fn'])==area
  paired.append(dict(id=sid,area_quartile=int(np.searchsorted(cuts,area,side='right'))+1,
   control_dice=float(a['dice']),drop_path_dice=float(b['dice']),delta=float(b['dice'])-float(a['dice']),
   control_recall=float(a['recall']),drop_path_recall=float(b['recall'])))
 delta=np.array([x['delta'] for x in paired])
 result=dict(overall=summary,paired_n=len(paired),mean_dice_delta=float(delta.mean()),
  improved=int((delta>0).sum()),worsened=int((delta<0).sum()),
  gains_over_5pp=int((delta>.05).sum()),losses_over_5pp=int((delta<-.05).sum()),
  severe_undercoverage={n:sum(float(x['recall'])<.5 for x in by[n].values()) for n in by},
  recurring_cases=[x for x in paired if x['id'] in ('305','581','791')],
  area_quartile_cuts_pixels=cuts.tolist(),
  area_quartiles={str(q):dict(n=sum(x['area_quartile']==q for x in paired),mean_delta=float(np.mean([x['delta'] for x in paired if x['area_quartile']==q]))) for q in range(1,5)},
  interpretation='One seed, repeated development validation. Primary endpoint is best validation macro Dice; diagnostic confidence metrics do not change selection.',test_evaluated=False)
 with (r/'paired_diagnosis.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 (r/'analysis.json').write_text(json.dumps(result,indent=2))
 (ROOT/'ANALYSIS_DONE.json').write_text(json.dumps(dict(complete=True,macro_dice_delta=result['mean_dice_delta'],test_evaluated=False),indent=2))
 print(json.dumps(dict(delta=result['mean_dice_delta'],severe_undercoverage=result['severe_undercoverage'])),flush=True)
if __name__=='__main__':main()
