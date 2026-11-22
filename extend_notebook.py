import os, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).parent
os.chdir(root)
os.environ['JUPYTER_PATH']=os.pathsep.join([str(Path(sys.prefix)/'share/jupyter'),str(Path(sys.base_prefix)/'share/jupyter')])
os.environ['JUPYTER_RUNTIME_DIR']=str(root/'.jupyter_runtime')
source=(root/'visualizations.py').read_text(encoding='utf-8')
cells=[]
for part in source.split('# %%')[1:]:
    if part.startswith(' [markdown]'):
        body=part.split('\n',1)[1]
        body='\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line for line in body.splitlines())
        cell=nbformat.v4.new_markdown_cell(body.strip())
    else:
        cell=nbformat.v4.new_code_cell(part.strip())
    cell.metadata['extended_analysis']=True
    cells.append(cell)
extra=nbformat.v4.new_notebook(cells=cells)
print('Executing extended visualizations using saved model...',flush=True)
NotebookClient(extra,timeout=600,kernel_name='shakespeare-rnn',resources={'metadata':{'path':str(root)}}).execute()
path=root/'Shakespeare_SimpleRNN.ipynb'
nb=nbformat.read(path,as_version=4)
nb.cells=[c for c in nb.cells if not c.metadata.get('extended_analysis')]+extra.cells
nbformat.validate(nb)
nbformat.write(nb,path)
from nbconvert import HTMLExporter
html,_=HTMLExporter(embed_images=True).from_notebook_node(nb)
(root/'Shakespeare_SimpleRNN.html').write_text(html,encoding='utf-8')
print('Updated executed notebook and HTML.',flush=True)
