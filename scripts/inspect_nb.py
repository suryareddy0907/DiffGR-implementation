import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('notebooks/01_replicate_ml1m.ipynb', 'r', encoding='utf-8'))
print(f'Total cells: {len(nb["cells"])}')

for i, cell in enumerate(nb['cells']):
    c_type = cell['cell_type']
    outputs = cell.get('outputs', [])
    text_out = []
    for out in outputs:
        if 'text' in out:
            text_out.extend(out['text'])
        elif 'data' in out and 'text/plain' in out['data']:
            text_out.extend(out['data']['text/plain'])
    out_str = ''.join(text_out).strip()
    if out_str:
        print(f'=== Cell {i} ({c_type}) Output ===')
        print(out_str)
        print('=' * 50)
