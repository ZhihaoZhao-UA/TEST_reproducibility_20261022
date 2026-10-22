"""Post-outcome descriptive control only; no independent confirmation."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent

def run():
 rr=json.loads((P/'neural_path_n96_results.json').read_text())['records'];cells=sorted({(r['signal'],r['sigma'],r['horizon']) for r in rr});rows=[]
 for r in rr:rows.append({k:r[k] for k in ['bank','signal','sigma','horizon','use_sure']}|{'control':r['risks']['last'] if r['use_sure'] else r['risks']['uniform'],'adaptive':r['risks']['adaptive'],'cv3':r['risks']['cv3']})
 means={k:sum(r[k] for r in rows)/len(rows) for k in ['control','adaptive','cv3']};diffs=[sum(r['adaptive']-r['control'] for r in rows if (r['signal'],r['sigma'],r['horizon'])==c)/32 for c in cells];out={'status':'Post-outcome descriptive control, no independent confirmation or new p-values. Reuses identical training-only branch decisions, replaces SURE prediction by last-checkpoint prediction on the SURE branch.','means':means,'control_vs_adaptive':{'wins':sum(v>1e-10 for v in diffs),'ties':sum(abs(v)<=1e-10 for v in diffs),'losses':sum(v< -1e-10 for v in diffs)},'records':rows};(P/'neural_path_n96_last_control.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':run()
