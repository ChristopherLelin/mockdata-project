import pandas as pd
import pickle
from scripts.extract_schema import get_connection, get_table_data_sample
from scripts.load_config import load_pattern_definitions

print('=== DEBUGGING PRIMARY KEY ISSUE ===')

# 1. Check sample data
engine = get_connection()
sample_data = get_table_data_sample(engine, 'T667_TCFIXNOTE')
max_val = pd.to_numeric(sample_data['FIXNOTE_NO'], errors='coerce').max()
print(f'Max FIXNOTE_NO in DB sample: {max_val}')
print(f'Next value should start from: {int(max_val) + 1}')

# 2. Check pattern definitions
pattern_defs = load_pattern_definitions('config/pattern_definitions.xlsx')
tcfixnote_pattern = pattern_defs[
    (pattern_defs['TableName'].str.upper() == 'T667_TCFIXNOTE') &
    (pattern_defs['ColumnName'].str.upper() == 'FIXNOTE_NO')
]
print(f'\nPattern found: {not tcfixnote_pattern.empty}')
if not tcfixnote_pattern.empty:
    print(f'Pattern type: {tcfixnote_pattern.iloc[0]["PatternType"]}')

# 3. Check generated data
with open('output/generated_data.pickle', 'rb') as f:
    data = pickle.load(f)

fixnote_data = data['T667_tcfixnote']
print(f'\nGenerated data:')
print(f'Min value: {fixnote_data["FIXNOTE_NO"].min()}')
print(f'Max value: {fixnote_data["FIXNOTE_NO"].max()}')
print(f'Unique count: {fixnote_data["FIXNOTE_NO"].nunique()} out of {len(fixnote_data)} total')

# 4. The problem: IncrementFromDB is not being applied!
print(f'\n=== CONCLUSION ===')
print(f'Expected: Values should start from {int(max_val) + 1} and be unique')
print(f'Actual: Values start from {fixnote_data["FIXNOTE_NO"].min()} with duplicates')
print(f'The IncrementFromDB pattern is not being applied correctly!')
