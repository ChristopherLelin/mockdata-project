"""
Script for inserting generated mock data into SQL Server.

This module reads the generated data from Excel and inserts it into
the SQL Server database, respecting table dependencies.
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from tqdm import tqdm
import logging

from extract_schema import get_connection, has_is_mock_column
from load_config import load_table_hierarchy, load_target_tables, determine_generation_order

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_generated_data(excel_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load generated data from Excel file.
    
    Args:
        excel_path: Path to the Excel file containing generated data
        
    Returns:
        Dictionary mapping table names to DataFrames with generated data
    """
    logger.info(f"Loading generated data from {excel_path}")
    
    # Get list of sheet names (table names)
    excel_file = pd.ExcelFile(excel_path)
    table_names = excel_file.sheet_names
    
    # Load each sheet into a DataFrame
    data_dict = {}
    for table_name in table_names:
        df = pd.read_excel(excel_file, sheet_name=table_name)
        data_dict[table_name] = df
        logger.info(f"Loaded {len(df)} rows for table {table_name}")
    
    return data_dict

def generate_sql_file(
    data_dict: Dict[str, pd.DataFrame], 
    output_path: str,
    insertion_order: Optional[List[str]] = None
):
    """
    Generate SQL inserts for all tables and save to a file.
    
    Args:
        data_dict: Dictionary mapping table names to DataFrames
        output_path: Path to save the SQL file
        insertion_order: Optional list specifying order of tables
    """
    logger.info(f"Generating SQL insert statements to {output_path}")
    
    # Use provided order or default to alphabetical
    if insertion_order is None:
        insertion_order = sorted(data_dict.keys())
    else:
        # Filter to only include tables in our data dictionary
        insertion_order = [t for t in insertion_order if t in data_dict]
    
    with open(output_path, 'w') as f:
        # Write header comment
        f.write("-- Generated Mock Data SQL Inserts\n")
        f.write(f"-- Generated tables: {', '.join(insertion_order)}\n\n")
        
        # Process each table in order
        for table_name in insertion_order:
            df = data_dict[table_name]
            
            if df.empty:
                continue
                
            # Write table header
            f.write(f"-- Table: {table_name} ({len(df)} rows)\n")
            
            # Generate INSERT statements in batches
            batch_size = 1000  # Process in batches to avoid huge SQL statements
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                # Get column names, handling special characters
                columns = [f"[{col}]" for col in batch_df.columns]
                column_list = ", ".join(columns)
                
                # Start the INSERT statement
                f.write(f"INSERT INTO [{table_name}] ({column_list})\nVALUES\n")
                
                # Generate value tuples
                rows = []
                for _, row in batch_df.iterrows():
                    values = []
                    for val in row:
                        if pd.isna(val):
                            values.append("NULL")
                        elif isinstance(val, (int, float, bool)):
                            values.append(str(val))
                        elif isinstance(val, (pd.Timestamp, pd.DatetimeTZDtype)):
                            values.append(f"'{val}'")
                        else:
                            # Escape single quotes in strings
                            val_str = str(val).replace("'", "''")
                            values.append(f"'{val_str}'")
                    
                    rows.append(f"({', '.join(values)})")
                
                # Join rows with commas
                f.write(",\n".join(rows))
                f.write(";\n\n")
            
            f.write("\n")
            
        # Write completion comment
        f.write("-- End of generated inserts\n")
    
    logger.info(f"SQL insert statements written to {output_path}")

def insert_into_database(
    data_dict: Dict[str, pd.DataFrame], 
    engine: Engine,
    insertion_order: Optional[List[str]] = None
):
    """
    Insert generated data directly into the database.
    
    Args:
        data_dict: Dictionary mapping table names to DataFrames
        engine: SQLAlchemy engine connected to the database
        insertion_order: Optional list specifying order of tables
    """
    # Use provided order or default to alphabetical
    if insertion_order is None:
        insertion_order = sorted(data_dict.keys())
    else:
        # Filter to only include tables in our data dictionary
        insertion_order = [t for t in insertion_order if t in data_dict]
    
    # Create a database connection
    connection = engine.connect()
    transaction = connection.begin()
    
    try:
        # Process each table in order
        for table_name in tqdm(insertion_order, desc="Inserting Tables"):
            df = data_dict[table_name]
            
            if df.empty:
                logger.info(f"Skipping empty table: {table_name}")
                continue
                
            logger.info(f"Inserting {len(df)} rows into table: {table_name}")
            
            # Check if IsMock column exists in the table
            needs_is_mock = 'IsMock' in df.columns and not has_is_mock_column(engine, table_name)
            
            # If the IsMock column doesn't exist in the database but is in our data,
            # remove it to avoid insertion errors
            if needs_is_mock:
                logger.warning(f"IsMock column not found in {table_name} table schema. Removing before insert.")
                df = df.drop(columns=['IsMock'])
            
            # Insert in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                
                # Convert numpy NaN values to None for SQL compatibility
                batch_df = batch_df.replace({np.nan: None})
                
                # Insert batch
                batch_df.to_sql(
                    table_name, 
                    connection, 
                    if_exists='append', 
                    index=False
                )
                
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1} for {table_name}")
        
        # Commit the transaction
        transaction.commit()
        logger.info("All data successfully inserted into the database!")
        
    except Exception as e:
        # Roll back in case of error
        transaction.rollback()
        logger.error(f"Error inserting data: {str(e)}")
        raise
    finally:
        # Close the connection
        connection.close()

def determine_insertion_order(generated_data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Determine the order for inserting tables based on the hierarchy.
    If table_hierarchy.xlsx file exists, use that, otherwise use alphabetical order.
    
    Args:
        generated_data: Dictionary mapping table names to DataFrames
        
    Returns:
        List of table names in insertion order
    """
    try:
        # Check if table_hierarchy.xlsx exists
        hierarchy_path = os.path.join('config', 'table_hierarchy.xlsx')
        if os.path.exists(hierarchy_path):
            # Load hierarchy information
            table_hierarchy = load_table_hierarchy(hierarchy_path)
            
            # Load target tables information to get the full list
            tables_path = os.path.join('config', 'target_tables.xlsx')
            if os.path.exists(tables_path):
                target_tables = load_target_tables(tables_path)
                
                # Determine order using the hierarchy
                order = determine_generation_order(
                    target_tables,
                    pd.DataFrame(columns=['ParentTable', 'ParentColumn', 'ChildTable', 'ChildColumn']),
                    table_hierarchy
                )
                
                # Filter to only include tables in our generated data
                return [t for t in order if t in generated_data]
    except Exception as e:
        logger.warning(f"Error determining insertion order from hierarchy: {str(e)}")
    
    # Default to alphabetical order if no hierarchy or error occurred
    return sorted(generated_data.keys())

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Insert generated mock data into SQL Server.")
    parser.add_argument("--source", required=True, help="Path to Excel file with generated data")
    parser.add_argument("--sql-output", default="output/generated_data.sql", help="Path to save SQL inserts")
    parser.add_argument("--insert", action="store_true", help="Directly insert data into database")
    
    return parser.parse_args()

def main():
    """Main entry point for the data insertion script."""
    args = parse_args()
    
    # Setup output directory for SQL file if it doesn't exist
    output_dir = os.path.dirname(args.sql_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load generated data from Excel
    generated_data = load_generated_data(args.source)
    
    # Determine insertion order
    insertion_order = determine_insertion_order(generated_data)
    
    # Generate SQL file
    generate_sql_file(generated_data, args.sql_output, insertion_order)
    
    # Insert into database if requested
    if args.insert:
        logger.info("Inserting data into database...")
        engine = get_connection()
        insert_into_database(generated_data, engine, insertion_order)
    else:
        logger.info("Data not inserted into database. Use --insert flag to enable database insertion.")
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()
