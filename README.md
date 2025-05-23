# 🚀 SQL Server Hierarchical Mock Data Generator

This tool generates realistic mock data for SQL Server tables, respecting schema constraints, primary/foreign keys, and hierarchical relationships.

## 🧩 Features

- **Schema-Aware Generation**
  - Automatically extracts table schema from SQL Server
  - Preserves primary key, foreign key, and column constraints
  - Enforces max column lengths, data types, and nullability

- **Data-Driven Modeling**
  - Uses SDV (Synthetic Data Vault) to learn from existing data
  - Supports text, numeric, and alphanumeric key patterns
  - Mimics distributions of real data

- **Hierarchical Integrity**
  - Follows parent-child relationships in correct order
  - Ensures valid foreign key generation
  - Automatically inserts data respecting dependencies

- **Customization & Traceability**
  - Add `IsMock = 1` to generated rows
  - Preview data in Excel before database insertion
  - Control generation via simple configuration files

## 📋 Requirements

- Python 3.8+
- SQL Server with ODBC Driver
- Required Python packages:
  ```
  pip install pandas openpyxl sqlalchemy pyodbc sdv python-dotenv tqdm
  ```

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file in the project root (see `.env.template` for an example)
3. Create or update the Excel configuration files in the `config` directory

### Configuration Files

The tool uses four Excel files to control data generation:

1. **`target_tables.xlsx`**: Tables to generate data for and row counts
2. **`fk_mappings.xlsx`**: Additional foreign key relationships not in the schema
3. **`unique_constraints.xlsx`**: Additional uniqueness requirements
4. **`table_hierarchy.xlsx`**: Generation order for hierarchical tables

You can create sample configurations by running:
```
python create_sample_configs.py
```

## 🚀 Usage

### Step 1: Generate Mock Data

```bash
python scripts/generate_data.py \
  --tables config/target_tables.xlsx \
  --fks config/fk_mappings.xlsx \
  --uniques config/unique_constraints.xlsx \
  --hierarchy config/table_hierarchy.xlsx
```

This will:
1. Connect to your SQL Server using credentials from the `.env` file
2. Extract schema information for the specified tables
3. Learn data patterns from existing rows
4. Generate mock data respecting all constraints
5. Save the data to `output/generated_data.xlsx`

### Step 2: Insert Data (Optional)

After reviewing the generated data in Excel, you can insert it into the database:

```bash
python scripts/insert_data.py \
  --source output/generated_data.xlsx \
  --insert
```

This will:
1. Read the generated data from Excel
2. Generate an SQL script at `output/generated_data.sql`
3. Insert the data into the database if `--insert` is specified

## 📁 Project Structure

```
├── config/                  # Configuration files
│   ├── target_tables.xlsx   # Tables and row counts
│   ├── fk_mappings.xlsx     # Additional foreign keys
│   ├── unique_constraints.xlsx # Uniqueness rules
│   └── table_hierarchy.xlsx # Table generation order
├── output/                  # Output files
│   ├── generated_data.xlsx  # Generated data in Excel
│   └── generated_data.sql   # SQL insert statements
├── scripts/                 # Python modules
│   ├── extract_schema.py    # DB schema extraction
│   ├── load_config.py       # Configuration loading
│   ├── generate_data.py     # Data generation logic
│   └── insert_data.py       # Database insertion
├── .env                     # Database credentials
└── .env.template            # Template for .env
```

## ⚙️ Customization

- **Modify Generation Logic**: Edit the `generate_data.py` file to customize how certain columns are generated
- **Change Output Format**: Adjust the `save_to_excel` method in `generate_data.py` to change output formats
- **Add Database Support**: Extend `extract_schema.py` to work with other database systems
