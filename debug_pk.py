import pandas as pd
import pickle
from scripts.extract_schema import get_connection, get_table_data_sample
from scripts.load_config import load_pattern_definitions

# Check sample data from database
engine = get_connection()
sample_data = get_table_data_sample(engine, 'T667_TCFIXNOTE')
print('Sample data from database:')
print(f'FIXNOTE_NO column exists: {"FIXNOTE_NO" in sample_data.columns}')
if 'FIXNOTE_NO' in sample_data.columns:
    print(f'Sample FIXNOTE_NO values: {sample_data["FIXNOTE_NO"].head(10).tolist()}')
    print(f'Max FIXNOTE_NO in sample: {pd.to_numeric(sample_data["FIXNOTE_NO"], errors="coerce").max()}')
else:
    print('Available columns:', sample_data.columns.tolist())

# Check pattern definitions
pattern_defs = load_pattern_definitions('config/pattern_definitions.xlsx')
print('\nPattern definitions:')
print(pattern_defs)

# Check what pattern is found for T667_tcfixnote
tcfixnote_patterns = pattern_defs[
    (pattern_defs['TableName'].str.lower() == 't667_tcfixnote') &
    (pattern_defs['ColumnName'].str.lower() == 'fixnote_no')
]
print(f'\nPattern for T667_tcfixnote.FIXNOTE_NO:')
print(tcfixnote_patterns)

# Check the actual issue - what happens in the pattern application
print('\n=== DEBUG: Pattern application issue ===')
print('The issue is likely in _apply_pattern_definitions method')
print('1. Pattern is found correctly')
print('2. But IncrementFromDB logic is not working')
print('3. Generated data shows values starting from 0, not 4068')
print('4. This suggests the case-sensitive comparison in pattern application')

# Add this at the end to check the data generation logic
print('\n=== DEBUG: Checking data_samples structure ===')
import sys
sys.path.append('scripts')
from generate_data import DataGenerator

# Create a data generator instance to check its data_samples
config_dir = 'config'
generator = DataGenerator(config_dir)
generator.load_schemas()

print(f'data_samples keys: {list(generator.data_samples.keys())}')
for key in generator.data_samples.keys():
    if 'tcfixnote' in key.lower():
        print(f'Key: {key}')
        print(f'Columns: {generator.data_samples[key].columns.tolist()}')
        if 'FIXNOTE_NO' in generator.data_samples[key].columns:
            max_val = pd.to_numeric(generator.data_samples[key]['FIXNOTE_NO'], errors='coerce').max()
            print(f'Max FIXNOTE_NO in {key}: {max_val}')
