from pathlib import Path
import nbformat

root = Path(__file__).parent
source = (root/'train_project.py').read_text(encoding='utf-8')
cells = []
for part in source.split('# %%')[1:]:
    if part.startswith(' [markdown]'):
        body = part.split('\n',1)[1]
        body = '\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line
                         for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(part.strip()))
nb = nbformat.v4.new_notebook(cells=cells,metadata={
    'kernelspec': {'display_name':'Shakespeare RNN (local GPU)','language':'python','name':'shakespeare-rnn'},
    'language_info': {'name':'python','version':'3.13.9'}})
nbformat.write(nb,root/'Shakespeare_SimpleRNN.ipynb')
print('Notebook created:',root/'Shakespeare_SimpleRNN.ipynb')
