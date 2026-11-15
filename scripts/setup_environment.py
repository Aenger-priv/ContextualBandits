"""Create a notebook environment using a resolved base Python executable."""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]

def main():
    base=Path(getattr(sys,'_base_executable',sys.executable)).resolve()
    directory=ROOT/'.venv-notebook'
    subprocess.run([str(base),'-m','venv',str(directory)],check=True)
    python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.run([str(python),'-m','pip','install','-e',str(ROOT)+'[notebook]'],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    print('Select',python,'as the notebook kernel.')

if __name__=='__main__':main()
