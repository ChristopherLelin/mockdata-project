import pandas as pd

try:
    print("=== Reading pattern_definitions.xlsx ===")
    df = pd.read_excel('config/pattern_definitions.xlsx')
    print("Columns:", df.columns.tolist())
    print("\nData:")
    print(df)
    
    print("\n=== Looking for T667_TCFIXNOTE/FIXNOTE_NO ===")
    # Check exact match
    exact = df[(df['TableName'] == 'T667_TCFIXNOTE') & (df['ColumnName'] == 'FIXNOTE_NO')]
    print(f"Exact match rows: {len(exact)}")
    if not exact.empty:
        print(exact)
    
    # Check case variations
    for table_variant in ['T667_TCFIXNOTE', 't667_tcfixnote', 'T667_tcfixnote']:
        for col_variant in ['FIXNOTE_NO', 'fixnote_no', 'Fixnote_No']:
            match = df[(df['TableName'] == table_variant) & (df['ColumnName'] == col_variant)]
            if not match.empty:
                print(f"Found match: Table={table_variant}, Column={col_variant}")
                print(match)

except Exception as e:
    print(f"Error: {e}")
