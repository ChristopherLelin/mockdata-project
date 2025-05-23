"""
Configuration loader module for the mock data generator.

This module handles loading configuration data from Excel files:
- target_tables.xlsx: Tables to generate data for and row counts
- fk_mappings.xlsx: Additional foreign key relationships
- unique_constraints.xlsx: Additional uniqueness requirements
- table_hierarchy.xlsx: Generation order for tables
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_target_tables(file_path: str) -> pd.DataFrame:
    """
    Load the target tables configuration from Excel.
    
    Args:
        file_path: Path to the target_tables.xlsx file
        
    Returns:
        DataFrame with columns: TableName, Include, NumRecords
    """
    try:
        df = pd.read_excel(file_path)
        required_columns = ['TableName', 'Include', 'NumRecords']
        
        # Validate required columns exist
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {file_path}")
        
        # Filter to only include tables marked as 'Yes'
        included_df = df[df['Include'].str.lower() == 'yes'].copy()
        
        logger.info(f"Loaded {len(included_df)} target tables from {file_path}")
        return included_df
    except Exception as e:
        logger.error(f"Error loading target tables from {file_path}: {str(e)}")
        raise

def load_fk_mappings(file_path: str) -> pd.DataFrame:
    """
    Load additional foreign key mappings from Excel.
    
    Args:
        file_path: Path to the fk_mappings.xlsx file
        
    Returns:
        DataFrame with columns: ParentTable, ParentColumn, ChildTable, ChildColumn
    """
    try:
        df = pd.read_excel(file_path)
        required_columns = ['ParentTable', 'ParentColumn', 'ChildTable', 'ChildColumn']
        
        # Validate required columns exist
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {file_path}")
        
        logger.info(f"Loaded {len(df)} additional foreign key mappings from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading FK mappings from {file_path}: {str(e)}")
        raise

def load_unique_constraints(file_path: str) -> pd.DataFrame:
    """
    Load additional uniqueness constraints from Excel.
    
    Args:
        file_path: Path to the unique_constraints.xlsx file
        
    Returns:
        DataFrame with columns: TableName, ColumnName
    """
    try:
        df = pd.read_excel(file_path)
        required_columns = ['TableName', 'ColumnName']
        
        # Validate required columns exist
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {file_path}")
        
        logger.info(f"Loaded {len(df)} additional unique constraints from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading unique constraints from {file_path}: {str(e)}")
        raise

def load_table_hierarchy(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load table hierarchy information from Excel.
    
    Args:
        file_path: Path to the table_hierarchy.xlsx file
        
    Returns:
        DataFrame with columns: TableName, ParentTable or None if file is empty
    """
    try:
        df = pd.read_excel(file_path)
        required_columns = ['TableName', 'ParentTable']
        
        # Validate required columns exist
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {file_path}")
        
        # Filter out rows with empty ParentTable values
        if not df.empty:
            df = df[df['ParentTable'].notna()]
        
        if df.empty:
            logger.info(f"No table hierarchy defined in {file_path}")
            return None
        else:
            logger.info(f"Loaded {len(df)} table hierarchy relationships from {file_path}")
            return df
    except Exception as e:
        logger.error(f"Error loading table hierarchy from {file_path}: {str(e)}")
        raise

def determine_generation_order(
    target_tables: pd.DataFrame, 
    fk_mappings: pd.DataFrame,
    table_hierarchy: Optional[pd.DataFrame]
) -> List[str]:
    """
    Determine the order in which tables should be generated based on dependencies.
    
    Args:
        target_tables: DataFrame of tables to generate
        fk_mappings: DataFrame of foreign key relationships
        table_hierarchy: Optional DataFrame with explicit hierarchical relationships
        
    Returns:
        List of table names in generation order
    """
    tables_to_generate = target_tables['TableName'].tolist()
    
    # Build dependency graph
    dependencies = {}
    for table in tables_to_generate:
        dependencies[table] = set()
    
    # Add dependencies from foreign key mappings
    for _, row in fk_mappings.iterrows():
        parent = row['ParentTable']
        child = row['ChildTable']
        
        # Only consider tables that are in our target list
        if parent in tables_to_generate and child in tables_to_generate:
            dependencies[child].add(parent)
    
    # Add dependencies from table hierarchy if provided
    if table_hierarchy is not None and not table_hierarchy.empty:
        for _, row in table_hierarchy.iterrows():
            child = row['TableName']
            parent = row['ParentTable']
            
            # Only consider tables that are in our target list
            if parent in tables_to_generate and child in tables_to_generate:
                dependencies[child].add(parent)
    
    # Perform topological sort
    generation_order = []
    visited = set()
    temp_visited = set()
    
    def visit(table):
        if table in temp_visited:
            raise ValueError(f"Circular dependency detected involving table: {table}")
        
        if table not in visited:
            temp_visited.add(table)
            
            for dependency in dependencies[table]:
                visit(dependency)
                
            temp_visited.remove(table)
            visited.add(table)
            generation_order.append(table)
    
    # Visit each table to build the generation order
    for table in tables_to_generate:
        if table not in visited:
            visit(table)
    
    logger.info(f"Determined generation order: {', '.join(generation_order)}")
    return generation_order
