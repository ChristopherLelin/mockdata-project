import pandas as pd
import os

# Create directory if it doesn't exist
os.makedirs('config', exist_ok=True)

# 1. Create target_tables.xlsx
target_tables = pd.DataFrame({
    'TableName': ['T065_SHIP', 'T667_tcfixnote', 'T578_VOYAGE'],
    'Include': ['Yes', 'Yes', 'Yes'],
    'NumRecords': [1000, 5000, 15000]
})
target_tables.to_excel('config/target_tables.xlsx', index=False)

# 2. Create fk_mappings.xlsx
fk_mappings = pd.DataFrame({
    'ParentTable': ['T065_SHIP', 'T065_SHIP'],
    'ParentColumn': ['NAME_SHIP', 'FixtureID'],
    'ChildTable': ['T578_VOYAGE', 'T667_tcfixnote'],
    'ChildColumn': ['ship', 'SHIP']
})
fk_mappings.to_excel('config/fk_mappings.xlsx', index=False)

# 3. Create unique_constraints.xlsx
unique_constraints = pd.DataFrame({
    'TableName': ['T065_SHIP'],
    'ColumnName': ['NAME_SHIP']
})
unique_constraints.to_excel('config/unique_constraints.xlsx', index=False)

# 4. Create table_hierarchy.xlsx
table_hierarchy = pd.DataFrame({
    'TableName': ['T667_tcfixnote', 'T578_VOYAGE'],
    'ParentTable': ['T065_SHIP', 'T065_SHIP']
})
table_hierarchy.to_excel('config/table_hierarchy.xlsx', index=False)

print("Sample configuration files created in 'config' directory.")
