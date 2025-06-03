import pandas as pd
import pickle

with open('output/generated_data.pickle', 'rb') as f:
    data = pickle.load(f)

fixnote_data = data['T667_tcfixnote']
print('FIXNOTE_NO column info:')
print('Total rows:', len(fixnote_data))
print('Unique FIXNOTE_NO values:', fixnote_data['FIXNOTE_NO'].nunique())
print('Min FIXNOTE_NO:', fixnote_data['FIXNOTE_NO'].min())
print('Max FIXNOTE_NO:', fixnote_data['FIXNOTE_NO'].max())
print('First 10 FIXNOTE_NO values:')
print(fixnote_data['FIXNOTE_NO'].head(10).tolist())
print('Value counts (showing duplicates):')
print(fixnote_data['FIXNOTE_NO'].value_counts().head(10))
