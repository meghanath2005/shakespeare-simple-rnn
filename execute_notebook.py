import os, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).parent
os.chdir(root)
os.environ['JUPYTER_PATH'] = os.pathsep.join([str(Path(sys.prefix)/'share/jupyter'), str(Path(sys.base_prefix)/'share/jupyter')])
os.environ['JUPYTER_RUNTIME_DIR'] = str(root/'.jupyter_runtime')
path = root/'Shakespeare_SimpleRNN.ipynb'
nb = nbformat.read(path,as_version=4)
def save(**kwargs):
    nbformat.write(nb,path)
client = NotebookClient(nb,timeout=7200,kernel_name='shakespeare-rnn',
                        resources={'metadata':{'path':str(root)}},on_cell_executed=save)
print('Starting Jupyter kernel...',flush=True)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Executed notebook saved.',flush=True)
