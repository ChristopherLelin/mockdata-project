import pandas as pd

# Quick check of pattern definitions
print("=== Pattern Definitions ===")
pattern_defs = pd.read_excel('config/pattern_definitions.xlsx')
print(pattern_defs)

print("\n=== Checking for T667_TCFIXNOTE patterns ===")
# Check exact case matching
exact_match = pattern_defs[
    (pattern_defs['TableName'] == 'T667_TCFIXNOTE') & 
    (pattern_defs['ColumnName'] == 'FIXNOTE_NO')
]
print(f"Exact match: {len(exact_match)} rows")
if not exact_match.empty:
    print(exact_match)

# Check case insensitive matching
case_insensitive = pattern_defs[
    (pattern_defs['TableName'].str.upper() == 'T667_TCFIXNOTE') & 
    (pattern_defs['ColumnName'].str.upper() == 'FIXNOTE_NO')
]
print(f"Case insensitive match: {len(case_insensitive)} rows")
if not case_insensitive.empty:
    print(case_insensitive)

# Check what values we actually have
print(f"\nActual TableName values: {pattern_defs['TableName'].unique()}")
print(f"Actual ColumnName values: {pattern_defs['ColumnName'].unique()}")
