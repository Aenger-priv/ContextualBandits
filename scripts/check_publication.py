"""Fail if report files, private artifacts or notebook outputs enter publication."""
from pathlib import Path
import re,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN=re.compile(r'(^|/)(research|private|records|archive|deliverables|results)(/|$)|\.(docx?|pdf|zip|pptx)$|(^|/)(IRP_|manuscript|bibliograph|citation-audit)',re.I)

def main():
    paths=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    bad=[p for p in paths if FORBIDDEN.search(p)]
    for p in paths:
        path=ROOT/p
        if path.suffix=='.ipynb':
            n=json.loads(path.read_text())
            if any(c.get('outputs') or c.get('execution_count') is not None for c in n['cells'] if c['cell_type']=='code'):
                bad.append(p+' (stored notebook outputs)')
        if path.suffix in {'.py','.ipynb','.md','.toml','.json'} and (str(Path.home())+'/') in path.read_text():bad.append(p+' (local private path)')
    # Inspect all new commit paths so deleting a document later cannot hide its history.
    commits=subprocess.check_output(['git','rev-list','HEAD','^a741f33ac6842e09bd3b7c48034280ac0ffac776'],cwd=ROOT,text=True).splitlines()
    for commit in commits:
        names=subprocess.check_output(['git','diff-tree','--root','--no-commit-id','--name-only','-r',commit],cwd=ROOT,text=True).splitlines()
        bad.extend(p+' (history)' for p in names if FORBIDDEN.search(p))
    if bad:raise SystemExit('Publication blocked: '+', '.join(sorted(set(bad))))
    print(f'Code-only publication audit passed: {len(paths)} tracked files, {len(commits)} new commits; no report paths or stored notebook outputs.')
if __name__=='__main__':main()
