import json

# Fix train_model.ipynb - move imports to top of cells
with open('notebooks/train_model.ipynb') as f:
    nb = json.load(f)

# Cell 22 (index 21): Move import to top of cell
cell_22 = nb['cells'][21]
source = cell_22['source']
imports = []
code_lines = []
for line in source:
    if line.strip().startswith('from egoindustrial.inference.tensorrt_engine import'):
        imports.append(line)
    else:
        code_lines.append(line)
cell_22['source'] = imports + code_lines

# Cell 24 (index 23): Move import json to top
cell_24 = nb['cells'][23]
source = cell_24['source']
imports = []
code_lines = []
for line in source:
    if line.strip().startswith('import json'):
        imports.append(line)
    else:
        code_lines.append(line)
cell_24['source'] = imports + code_lines

with open('notebooks/train_model.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed train_model.ipynb')
