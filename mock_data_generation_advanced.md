
# 🚀 Advanced Mock Data Generation Guide for SQL Server

## 🎯 Objective

To generate **hierarchical, constraint-respecting, and realistic** mock data based on existing SQL Server data and schema patterns.

This utility allows you to:
- Multiply your data while preserving **relational integrity**
- Learn and adapt to **custom data patterns** (like alphanumeric keys)
- Respect **column constraints** such as length, uniqueness, and types
- Preview generated data in **Excel** before optionally inserting into the database

---

## 🧩 Requirements

- **Python 3.8+**
- Required libraries:
  ```bash
  pip install pandas openpyxl sqlalchemy pyodbc sdv python-dotenv tqdm
  ```

---

## 📂 Input Files (All Excel Format)

### 1. `target_tables.xlsx`
Defines which tables need mock data and how many rows.

| TableName       | Include | NumRecords |
|---------------  |---------|------------|
| T065_SHIP       | Yes     | 1000       |
| T667_tcfixnote  | Yes     | 5000        |
| T578_VOYAGE     | Yes     | 15000        |

---

### 2. `fk_mappings.xlsx`
Lists foreign key relationships not defined in the schema.

| ParentTable | ParentColumn | ChildTable    | ChildColumn |
|-------------|--------------|------------   |-------------|
| T065_SHIP   | NAME_SHIP    | T578_VOYAGE   | ship        |
| T065_SHIP   | FixtureID    | T667_tcfixnote| SHIP        |

---

### 3. `unique_constraints.xlsx`
Lists additional column-level uniqueness requirements.

| TableName   | ColumnName     |
|-------------|----------------|
| T065_SHIP   | NAME_SHIP      |

---

### 4. `table_hierarchy.xlsx`
Defines generation order to ensure hierarchical dependency.

| TableName   | ParentTable |
|-------------|-------------|


---

## 🧠 Core Features

1. **Schema-Aware Generation**  
   - Unique values for primary keys
   - Enforces max column lengths, data types, and nullability
   - Additional constraints from `unique_constraints.xlsx`

2. **Data-Driven Modeling**  
   - Learns from existing data using SDV
   - Supports text and alphanumeric key patterns
   - Mimics distributions of real data

3. **Hierarchical Integrity**  
   - Follows parent-child relationships in order (ships → fixtures → estimates → voyages)
   - Ensures valid foreign key generation
   - Automatically inserts mock data in dependency order

4. **Traceability**  
   - Adds `IsMock = 1` to generated rows
   - Saves data preview in `generated_data.xlsx`

---

## 📤 Output

- `output/generated_data.xlsx`: Each table in a separate sheet
- `output/generated_data.sql`: Optional SQL inserts
- Optional: Direct DB insert using `--insert`

---

## 🔁 Workflow

1. Read schema + FK mappings + table hierarchy
2. Read `target_tables.xlsx` for row counts
3. Read uniqueness constraints
4. Learn patterns using SDV from current data
5. Generate hierarchical mock data and save to Excel
6. After validation, run with `--insert` to push to SQL Server

---

## 🚀 Execution Command

```bash
# Step 1: Generate mock data only
python scripts/generate_data.py \
  --tables config/target_tables.xlsx \
  --fks config/fk_mappings.xlsx \
  --uniques config/unique_constraints.xlsx \
  --hierarchy config/table_hierarchy.xlsx

# Step 2: Insert after review
python scripts/insert_data.py \
  --source output/generated_data.xlsx \
  --insert
```

---

## 🔐 DB Configuration

Create a `.env` file:

```env
DB_SERVER=localhost
DB_NAME=MyDatabase
DB_USER=sa
DB_PASSWORD=your_password
```

---

## 📎 Optional Enhancements

- Detect surrogate vs natural keys
- Anonymize PII fields
- Visualize FK dependencies with a graph
