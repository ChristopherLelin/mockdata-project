"""
Mock data generation module using SDV to create realistic, constraint-respecting data.

This module uses the Synthetic Data Vault (SDV) library to learn patterns from
existing data and generate new synthetic data that respects schema constraints,
primary/foreign keys, and uniqueness requirements.
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Tuple, Optional
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import FixedCombinations
from tqdm import tqdm
import logging
import random
import string
import re
import datetime
import pickle

from extract_schema import (
    get_connection,
    get_table_schema,
    get_primary_keys,
    get_foreign_keys,
    get_table_data_sample,
    analyze_pk_patterns
)
from load_config import (
    load_target_tables,
    load_fk_mappings,
    load_unique_constraints,
    load_table_hierarchy,
    determine_generation_order
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataGenerator:
    """Class for generating realistic mock data based on schema and patterns."""
    
    def __init__(self, config_dir: str):
        """
        Initialize the data generator.
        
        Args:
            config_dir: Directory containing configuration Excel files
        """
        self.config_dir = config_dir
        self.engine = get_connection()
        
        # Load configuration files
        self.target_tables = load_target_tables(os.path.join(config_dir, 'target_tables.xlsx'))
        self.fk_mappings = load_fk_mappings(os.path.join(config_dir, 'fk_mappings.xlsx'))
        self.unique_constraints = load_unique_constraints(os.path.join(config_dir, 'unique_constraints.xlsx'))
        # Load pattern definitions for advanced pattern-driven generation
        from load_config import load_pattern_definitions
        self.pattern_definitions = load_pattern_definitions(os.path.join(config_dir, 'pattern_definitions.xlsx'))
        
        # Load table hierarchy if available
        try:
            self.table_hierarchy = load_table_hierarchy(os.path.join(config_dir, 'table_hierarchy.xlsx'))
        except Exception:
            logger.warning("Table hierarchy file not found or invalid. Using FK-based ordering.")
            self.table_hierarchy = None
        
        # Determine generation order
        self.generation_order = determine_generation_order(
            self.target_tables, 
            self.fk_mappings, 
            self.table_hierarchy
        )
        
        # Store schema information and generated data
        self.table_schemas = {}
        self.primary_keys = {}
        self.foreign_keys = {}
        self.pk_patterns = {}
        self.data_samples = {}
        self.generated_data = {}
        self.all_unique_values = {}
        # Initialize column_patterns for pattern learning
        self.column_patterns = {}
        
    def load_schemas(self):
        """Load schema information for all target tables."""
        for table_name in self.generation_order:
            logger.info(f"Loading schema for table: {table_name}")
            
            # Get table schema
            self.table_schemas[table_name] = get_table_schema(self.engine, table_name)
            
            # Get primary keys
            self.primary_keys[table_name] = get_primary_keys(self.engine, table_name)
            
            # Get foreign keys
            self.foreign_keys[table_name] = get_foreign_keys(self.engine, table_name)
            
            # Get sample data for learning patterns
            self.data_samples[table_name] = get_table_data_sample(self.engine, table_name)
            
            # Analyze primary key patterns
            if self.primary_keys[table_name]:
                self.pk_patterns[table_name] = analyze_pk_patterns(
                    self.data_samples[table_name], 
                    self.primary_keys[table_name]
                )
            
            # Initialize storage for unique values
            for col in self.table_schemas[table_name]['column_name']:
                self.all_unique_values[(table_name, col)] = set()
                
                # Add existing values from sample data to track uniqueness
                if col in self.data_samples[table_name].columns:
                    self.all_unique_values[(table_name, col)].update(
                        self.data_samples[table_name][col].dropna().unique()                    )
            # Analyze and store column patterns for this table
            self._analyze_all_column_patterns(table_name)
    
    def _analyze_all_column_patterns(self, table_name: str):
        """
        Analyze data patterns for all columns in a table.
        
        This method examines sample data for each column to identify:
        - String patterns (e.g., prefixes, formats)
        - Numeric patterns (sequential, grouped increments)
        - Date/time patterns
        - Composite patterns
        - Column value distributions
        
        The identified patterns are stored for use during data generation.
        """
        if table_name not in self.data_samples or self.data_samples[table_name].empty:
            logger.warning(f"No sample data available for table {table_name} to analyze patterns")
            return
            
        # Get sample data and schema for this table
        sample_data = self.data_samples[table_name]
        schema_df = self.table_schemas[table_name]
        
        # Dictionary to store column patterns
        self.column_patterns = getattr(self, 'column_patterns', {})
        self.column_patterns[table_name] = {}
        
        # Process each column
        for _, col_info in schema_df.iterrows():
            column_name = col_info['column_name']
            data_type = col_info['data_type']
            
            # Skip if column not in sample data
            if column_name not in sample_data.columns:
                continue
                
            # Get non-null values for this column
            values = sample_data[column_name].dropna()
            if len(values) == 0:
                continue
                
            # Store pattern information
            pattern_info = {
                'data_type': data_type,
                'pattern_type': 'unknown',
                'value_distribution': {},
                'unique': False
            }
            
            # Check if column has unique values
            if len(values.unique()) == len(values):
                pattern_info['unique'] = True
                
            # Analyze patterns based on data type
            if data_type in ('int', 'bigint', 'smallint', 'tinyint', 'decimal', 'numeric'):
                # Try to convert to numeric for analysis
                try:
                    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                    if len(numeric_values) > 0:
                        # Check if values are sequential
                        sorted_vals = sorted(numeric_values)
                        if len(sorted_vals) > 1 and all(sorted_vals[i+1] - sorted_vals[i] == 1 for i in range(len(sorted_vals)-1)):
                            pattern_info['pattern_type'] = 'sequential'
                        
                        # Check if values follow other numeric patterns
                        # Example: grouped increments, multiples, etc.
                        
                        # Store value distribution statistics
                        pattern_info['value_distribution'] = {
                            'min': float(numeric_values.min()),
                            'max': float(numeric_values.max()),
                            'mean': float(numeric_values.mean()),
                            'median': float(numeric_values.median())
                        }
                except Exception as e:
                    logger.debug(f"Error analyzing numeric pattern for {table_name}.{column_name}: {e}")
                    
            elif data_type in ('char', 'varchar', 'nchar', 'nvarchar'):
                # String pattern analysis
                string_values = values.astype(str)
                
                # Check for common prefixes
                if len(string_values) > 0:
                    # Get prefixes of length 1-3 characters
                    prefix_counts = {}
                    for prefix_len in range(1, 4):
                        prefixes = string_values.str[:prefix_len].value_counts()
                        if prefixes.max() / len(string_values) > 0.5:  # If more than 50% share prefix
                            prefix_counts[prefix_len] = prefixes.idxmax()
                    
                    if prefix_counts:
                        # Use the longest common prefix
                        max_len = max(prefix_counts.keys())
                        pattern_info['pattern_type'] = f'prefixed_{prefix_counts[max_len]}'
                        
                # Check for regex patterns
                # TODO: Implement regex pattern detection
                        
            elif data_type in ('datetime', 'date', 'time'):
                # Date/time pattern analysis
                # TODO: Implement date/time pattern detection
                pass
                
            # Store the pattern information
            self.column_patterns[table_name][column_name] = pattern_info
            
        # Look for relationships between columns in the same table
        # TODO: Implement inter-column relationship detection
        
        logger.debug(f"Completed pattern analysis for table: {table_name}")

    def generate_all_data(self):
        """Generate mock data for all tables in the correct order."""
        for table_name in tqdm(self.generation_order, desc="Generating Tables"):
            logger.info(f"Generating data for table: {table_name}")
            
            # Get number of records to generate
            num_records = self.target_tables.loc[
                self.target_tables['TableName'] == table_name, 'NumRecords'
            ].iloc[0]
            
            # Generate data for this table
            self.generated_data[table_name] = self.generate_table_data(table_name, int(num_records))
            
            # Update unique value tracking for this table
            for col in self.generated_data[table_name].columns:
                if (table_name, col) in self.all_unique_values:
                    self.all_unique_values[(table_name, col)].update(
                        self.generated_data[table_name][col].dropna().unique()
                    )
    
    def generate_table_data(self, table_name: str, num_records: int) -> pd.DataFrame:
        """
        Generate synthetic data for a specific table.
        
        Args:
            table_name: Name of the table to generate data for
            num_records: Number of records to generate
            
        Returns:
            DataFrame containing the generated data
        """
        # Get schema information
        schema_df = self.table_schemas[table_name]
        pk_columns = self.primary_keys[table_name]
        sample_data = self.data_samples[table_name].copy()
        
        # Convert data types to ensure numeric columns are properly processed
        for _, col_info in schema_df.iterrows():
            column_name = col_info['column_name']
            data_type = col_info['data_type']
            
            # Skip if column not in sample_data
            if column_name not in sample_data.columns:
                continue
                
            # Convert numeric types to appropriate data types
            if data_type in ('int', 'bigint', 'smallint', 'tinyint'):
                # Try to convert to numeric, but handle errors gracefully
                try:
                    sample_data[column_name] = pd.to_numeric(sample_data[column_name], errors='coerce')
                except:
                    logger.warning(f"Could not convert column {column_name} to numeric")
            elif data_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                try:
                    sample_data[column_name] = pd.to_numeric(sample_data[column_name], errors='coerce')
                except:
                    logger.warning(f"Could not convert column {column_name} to float")
        
        # Setup SDV synthesizer with appropriate metadata
        metadata = self._create_metadata(table_name, schema_df)
        
        # Update metadata with constraints
        self._create_constraints(table_name, metadata)
        
        # For SDV 1.12.0, we need to use the updated API
        from sdv.single_table import GaussianCopulaSynthesizer
        
        synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata
        )
          # Fit the model on sample data
        logger.info(f"Training synthesizer on {len(sample_data)} samples for table: {table_name}")
        synthesizer.fit(sample_data)
        
        # Generate initial synthetic data
        logger.info(f"Generating {num_records} records for table: {table_name}")
        synthetic_data = synthesizer.sample(num_records)
        
        # Post-process to ensure all constraints are met
        processed_data = self._post_process_data(table_name, synthetic_data)
        # Apply learned column patterns to all columns in this table
        processed_data = self._apply_column_patterns(table_name, processed_data)
        # Apply pattern_definitions.xlsx rules
        processed_data = self._apply_pattern_definitions(table_name, processed_data)
        return processed_data
    
    def _create_metadata(self, table_name: str, schema_df: pd.DataFrame) -> Dict[str, Any]:
        """Create metadata dictionary for SDV based on table schema."""
        # For SDV 1.12.0, we need to use the SDV SingleTableMetadata class
        from sdv.metadata import SingleTableMetadata
        
        metadata = SingleTableMetadata()
        
        # Get primary keys for special handling
        pk_columns = set(self.primary_keys[table_name])
        # First pass: Add all columns with their proper SDV types based on schema
        for _, row in schema_df.iterrows():
            column_name = row['column_name']
            data_type = row['data_type']
            
            # Map SQL Server types to SDV types
            if column_name in pk_columns:
                # For primary keys, just mark as id without additional parameters
                # SDV 1.12.0 doesn't support computer_representation for 'id' type anymore
                metadata.add_column(column_name, sdtype='id')
            elif data_type in ('int', 'bigint', 'smallint', 'tinyint'):
                metadata.add_column(column_name, sdtype='numerical', computer_representation='Int64')
            elif data_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                metadata.add_column(column_name, sdtype='numerical', computer_representation='Float')
            elif data_type in ('date', 'datetime', 'datetime2', 'smalldatetime'):
                metadata.add_column(column_name, sdtype='datetime')
            elif data_type in ('bit'):
                metadata.add_column(column_name, sdtype='boolean')
            elif data_type in ('char', 'varchar', 'nchar', 'nvarchar'):                # Handle string types with length information if available
                max_length = row.get('max_length', 0)
                if data_type.startswith('n'):  # Unicode strings
                    max_length = max_length // 2
                # Just add as categorical, we'll handle length constraints in post-processing
                metadata.add_column(column_name, sdtype='categorical')
            else:
                metadata.add_column(column_name, sdtype='categorical')
        
        # After registering all columns, now set the primary key
        for pk in pk_columns:
            metadata.set_primary_key(pk)
            
        return metadata
    
    def _create_constraints(self, table_name: str, metadata) -> None:
        """Update metadata with foreign key relationships for SDV 1.12.0."""
        # Add foreign key constraints from database schema
        fk_df = self.foreign_keys[table_name]
        for _, row in fk_df.iterrows():
            parent_table = row['parent_table']
            parent_column = row['parent_column']
            child_column = row['child_column']
            
            # Only add constraint if parent table has been generated
            if parent_table in self.generated_data:
                logger.info(f"Foreign key relationship: {table_name}.{child_column} -> {parent_table}.{parent_column}")
                # Note this relationship for post-processing
                if not hasattr(self, 'fk_relationships'):
                    self.fk_relationships = []
                self.fk_relationships.append({
                    'child_table': table_name,
                    'child_column': child_column,
                    'parent_table': parent_table,
                    'parent_column': parent_column
                })
        
        # Add constraints from additional FK mappings
        additional_fks = self.fk_mappings[
            (self.fk_mappings['ChildTable'] == table_name)
        ]
        
        for _, row in additional_fks.iterrows():
            parent_table = row['ParentTable']
            parent_column = row['ParentColumn']
            child_column = row['ChildColumn']
            # Only add constraint if parent table has been generated
            if parent_table in self.generated_data:
                logger.info(f"Additional FK relationship: {table_name}.{child_column} -> {parent_table}.{parent_column}")
                # No fk_relationships logic here; just log or update metadata if needed
        
    def _post_process_data(self, table_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process generated data to ensure all constraints are met.
        
        This includes:
        - Ensuring primary keys are unique
        - Respecting column length limits
        - Handling special column types
        - Enforcing additional uniqueness constraints
        - Removing the IsMock column if present
        """
        # Make a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Remove IsMock column if it exists
        if 'IsMock' in processed_data.columns:
            logger.info(f"Removing IsMock column from table: {table_name}")
            processed_data = processed_data.drop(columns=['IsMock'])
            
        schema_df = self.table_schemas[table_name]
        pk_columns = self.primary_keys[table_name]
        
        # Process each column
        for _, col_info in schema_df.iterrows():
            column_name = col_info['column_name']
            
            # Skip if column not in generated data
            if column_name not in processed_data.columns:
                continue
            
            # Handle primary key columns
            if column_name in pk_columns:
                processed_data = self._ensure_unique_primary_key(
                    table_name, 
                    processed_data, 
                    column_name
                )
            
            # Handle string length constraints
            if col_info['data_type'] in ('char', 'varchar', 'nchar', 'nvarchar'):
                max_length = col_info['max_length']
                # For nchar/nvarchar, the length is in bytes, but each character takes 2 bytes
                if col_info['data_type'] in ('nchar', 'nvarchar'):
                    max_length = max_length // 2
                    
                processed_data = self._enforce_string_length(
                    processed_data, 
                    column_name, 
                    max_length
                )
            
            # Handle additional uniqueness constraints
            unique_cols = self.unique_constraints[
                (self.unique_constraints['TableName'] == table_name) & 
                (self.unique_constraints['ColumnName'] == column_name)
            ]
            
            if not unique_cols.empty:
                processed_data = self._ensure_unique_values(
                    table_name,
                    processed_data,
                    column_name
                )
        
        # Handle foreign key relationships
        processed_data = self._enforce_foreign_keys(table_name, processed_data)
        
        return processed_data
    
    def _ensure_unique_primary_key(
        self, 
        table_name: str, 
        data: pd.DataFrame, 
        column_name: str
    ) -> pd.DataFrame:
        # If pattern_definitions.xlsx says IncrementFromDB, enforce strict integer sequence and uniqueness
        # Use case-insensitive comparison for table and column names
        pattern_row = self.pattern_definitions[
            (self.pattern_definitions['TableName'].str.upper() == table_name.upper()) &
            (self.pattern_definitions['ColumnName'].str.upper() == column_name.upper()) &
            (self.pattern_definitions['PatternType'] == 'IncrementFromDB')
        ]
        if not pattern_row.empty:
            try:
                # Find the actual table name in data_samples (case-insensitive)
                actual_table_name = None
                for sample_table in self.data_samples.keys():
                    if sample_table.upper() == table_name.upper():
                        actual_table_name = sample_table
                        break
                
                # Find the actual column name (case-insensitive)
                actual_col_name = None
                if actual_table_name:
                    for col in self.data_samples[actual_table_name].columns:
                        if col.upper() == column_name.upper():
                            actual_col_name = col
                            break
                
                if actual_table_name and actual_col_name:
                    max_val = pd.to_numeric(self.data_samples[actual_table_name][actual_col_name], errors='coerce').max()
                    if pd.isna(max_val):
                        max_val = 0
                else:
                    max_val = 0
            except Exception:
                max_val = 0
            start_val = int(max_val) + 1
            # Always assign as integer sequence, guarantee uniqueness
            data[column_name] = range(start_val, start_val + len(data))
            data[column_name] = data[column_name].astype(int)
            return data.drop_duplicates(subset=[column_name]).reset_index(drop=True)
        
        # Ensure PKs are unique across both sample and generated data, and respect max_length
        result_data = data.copy()
        pattern = self.pk_patterns[table_name].get(column_name, 'unknown')
        # Get max_length for this PK column from schema
        schema_df = self.table_schemas[table_name]
        col_info = schema_df[schema_df['column_name'] == column_name].iloc[0]
        max_length = col_info.get('max_length', None)
        if col_info['data_type'] in ('nchar', 'nvarchar') and max_length:
            max_length = max_length // 2
        # Combine all unique values from sample and generated data
        # Find the actual table and column names (case-insensitive)
        actual_table_name = None
        for sample_table in self.data_samples.keys():
            if sample_table.upper() == table_name.upper():
                actual_table_name = sample_table
                break
        
        actual_col_name = None
        if actual_table_name:
            for col in self.data_samples[actual_table_name].columns:
                if col.upper() == column_name.upper():
                    actual_col_name = col
                    break
        
        if actual_table_name and actual_col_name:
            existing_values = set(self.data_samples[actual_table_name][actual_col_name].dropna().unique())
        else:
            existing_values = set()
        existing_values.update(self.all_unique_values.get((table_name, column_name), set()))
        for idx in result_data.index:
            current_value = result_data.at[idx, column_name]
            if pd.isna(current_value) or current_value in existing_values or (max_length and isinstance(current_value, str) and len(current_value) > max_length):
                if pattern == 'numeric' and pd.api.types.is_numeric_dtype(result_data[column_name].dtype):
                    max_val = max([v for v in existing_values if isinstance(v, (int, float))], default=0)
                    new_value = int(max_val) + 1
                else:
                    prefix = ''
                    suffix_len = max_length if max_length else 8
                    # Try to use a prefix if pattern is prefixed_*
                    if pattern.startswith('prefixed_'):
                        prefix = pattern.split('_', 1)[1]
                        suffix_len = max((max_length or 8) - len(prefix), 1)
                    elif pattern == 'unknown' and max_length:
                        prefix = ''
                        suffix_len = max_length
                    else:
                        # Use table abbreviation as prefix if possible
                        prefix = table_name[:3]
                        suffix_len = max((max_length or 8) - len(prefix), 1)
                    tries = 0
                    while True:
                        # Use digits for suffix if possible, else alphanum
                        if suffix_len > 0:
                            if tries < 10000:
                                suffix = ''.join(random.choices(string.digits, k=suffix_len))
                            else:
                                suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=suffix_len))
                        else:
                            suffix = ''
                        new_value = f"{prefix}{suffix}"[:max_length] if max_length else f"{prefix}{suffix}"
                        if new_value not in existing_values and (not max_length or len(new_value) <= max_length):
                            break
                        tries += 1
                        if tries > 100000:
                            raise Exception(f"Could not generate unique PK for {table_name}.{column_name} within max_length {max_length}")
                result_data.at[idx, column_name] = new_value
                existing_values.add(new_value)
        self.all_unique_values[(table_name, column_name)] = existing_values
        return result_data

    def _apply_pattern_definitions(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rules from pattern_definitions.xlsx to the generated DataFrame for this table.
        """
        patterns = self.pattern_definitions
        # Use case-insensitive comparison for table names
        table_patterns = patterns[patterns['TableName'].str.upper() == table_name.upper()]
        
        # If no patterns for this table, return unchanged
        if table_patterns.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply each pattern type
        for _, pattern_row in table_patterns.iterrows():
            col_name = pattern_row['ColumnName']
            pattern_type = pattern_row['PatternType']
            # Use GroupByColumn for grouped patterns, PatternFormat for composite patterns
            group_by_column = pattern_row.get('GroupByColumn')
            pattern_format = pattern_row.get('PatternFormat')
            
            # Skip if column isn't in the dataframe (case-insensitive check)
            matching_cols = [col for col in result_df.columns if col.upper() == col_name.upper()]
            if not matching_cols:
                continue
            actual_col_name = matching_cols[0]  # Use the actual column name from the dataframe
                
            # Handle IncrementFromDB pattern - generate integer values starting from max DB value + 1
            if pattern_type == 'IncrementFromDB':
                try:
                    # Find the actual table name in data_samples (case-insensitive)
                    actual_table_name = None
                    for sample_table in self.data_samples.keys():
                        if sample_table.upper() == table_name.upper():
                            actual_table_name = sample_table
                            break
                    
                    if actual_table_name and actual_col_name in self.data_samples[actual_table_name].columns:
                        max_val = pd.to_numeric(self.data_samples[actual_table_name][actual_col_name], errors='coerce').max()
                        if pd.isna(max_val):
                            max_val = 0
                    else:
                        max_val = 0
                except Exception:
                    max_val = 0
                start_val = int(max_val) + 1
                result_df[actual_col_name] = range(start_val, start_val + len(result_df))
                result_df[actual_col_name] = result_df[actual_col_name].astype(int)
                
            # Handle GroupedIncrement pattern - generate sequential numbers within groups
            elif pattern_type == 'GroupedIncrement':
                # Check if group_by_column exists (case-insensitive)
                group_col_matches = [col for col in result_df.columns if col.upper() == group_by_column.upper()] if group_by_column else []
                if group_col_matches:
                    actual_group_col = group_col_matches[0]
                    # Get current max value per group from sample data
                    group_maxes = {}
                    
                    # Find the actual table name in data_samples (case-insensitive)
                    actual_table_name = None
                    for sample_table in self.data_samples.keys():
                        if sample_table.upper() == table_name.upper():
                            actual_table_name = sample_table
                            break
                    
                    if actual_table_name and actual_group_col in self.data_samples[actual_table_name] and actual_col_name in self.data_samples[actual_table_name]:
                        sample_df = self.data_samples[actual_table_name]
                        grouped = sample_df.groupby(actual_group_col)[actual_col_name].max()
                        for group, max_val in grouped.items():
                            if pd.notna(max_val):
                                group_maxes[group] = int(max_val)
                    
                    # Apply increment within each group
                    for group in result_df[actual_group_col].unique():
                        mask = result_df[actual_group_col] == group
                        start = group_maxes.get(group, 0) + 1
                        group_size = mask.sum()
                        result_df.loc[mask, actual_col_name] = range(start, start + group_size)
                    
                    result_df[actual_col_name] = result_df[actual_col_name].astype(int)
                
            # Handle CompositePattern - generate values using a format string
            elif pattern_type == 'CompositePattern':
                if pattern_format:
                    # Define a function to apply the pattern
                    def composite(row):
                        try:
                            return pattern_format.format(**row.to_dict())
                        except Exception as e:
                            logger.warning(f"Failed to apply composite pattern: {e}")
                            return row[actual_col_name]
                    
                    # Apply the function to each row
                    result_df[actual_col_name] = result_df.apply(composite, axis=1)
        
        return result_df
        
    def _apply_column_patterns(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned column patterns to the generated data.
        
        This method uses the patterns detected in _analyze_all_column_patterns to 
        modify the generated data to better match the observed patterns in the sample data.
        
        Args:
            table_name: Name of the table being processed
            df: DataFrame containing the generated data
            
        Returns:
            DataFrame with column patterns applied
        """
        # Check if we have patterns for this table
        if table_name not in self.column_patterns:
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Get column patterns for this table
        patterns = self.column_patterns[table_name]
        
        # Apply patterns to each column
        for column_name, pattern_info in patterns.items():
            # Skip if column not in dataframe
            if column_name not in result_df.columns:
                continue
                
            pattern_type = pattern_info.get('pattern_type', 'unknown')
            
            # Apply different transformations based on pattern type
            if pattern_type == 'sequential':
                # For sequential patterns, generate sequential values
                if pd.api.types.is_numeric_dtype(result_df[column_name].dtype):
                    # Get min value to start sequence
                    min_val = pattern_info.get('value_distribution', {}).get('min', 0)
                    # Create sequential values
                    result_df[column_name] = range(int(min_val), int(min_val) + len(result_df))
                    
            elif pattern_type.startswith('prefixed_'):
                # For prefixed patterns, ensure all values have the prefix
                prefix = pattern_type.split('_', 1)[1]
                
                # Only process string columns
                if pd.api.types.is_string_dtype(result_df[column_name].dtype):
                    # Add prefix to values that don't already have it
                    mask = ~result_df[column_name].str.startswith(prefix, na=False)
                    result_df.loc[mask, column_name] = prefix + result_df.loc[mask, column_name].astype(str)
            
            # Handle other pattern types as needed
            # You can add more pattern handling here based on what patterns you expect to find
            
        return result_df
    
    def _ensure_unique_values(self, table_name: str, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Ensure values in a column are unique according to unique constraints.
        Similar to _ensure_unique_primary_key but for non-PK columns with uniqueness requirements.
        """
        # Make a copy to avoid modifying the original
        result_data = data.copy()
        
        # Get the column data type and max length
        schema_df = self.table_schemas[table_name]
        col_info = schema_df[schema_df['column_name'] == column_name].iloc[0]
        data_type = col_info['data_type']
        max_length = col_info.get('max_length', None)
        if data_type in ('nchar', 'nvarchar') and max_length:
            max_length = max_length // 2
            
        # Get existing unique values from sample data
        existing_values = set()
        if table_name in self.data_samples and column_name in self.data_samples[table_name].columns:
            existing_values.update(self.data_samples[table_name][column_name].dropna().unique())
        
        # Add values we've already seen in this generation run
        existing_values.update(self.all_unique_values.get((table_name, column_name), set()))
        
        # Check for patterns in pattern_definitions.xlsx with case-insensitive comparison
        pattern_row = self.pattern_definitions[
            (self.pattern_definitions['TableName'].str.upper() == table_name.upper()) &
            (self.pattern_definitions['ColumnName'].str.upper() == column_name.upper())
        ]
        
        # If using IncrementFromDB pattern, just return the data as it's already unique
        if not pattern_row.empty and pattern_row.iloc[0]['PatternType'] == 'IncrementFromDB':
            return result_data
            
        # Process each row, fixing non-unique values
        for idx in result_data.index:
            current_value = result_data.at[idx, column_name]
            
            # Skip if value is NA
            if pd.isna(current_value):
                continue
                
            # If value already exists in our set of known values, or exceeds max length, replace it
            if current_value in existing_values or (max_length and isinstance(current_value, str) and len(current_value) > max_length):
                if pd.api.types.is_numeric_dtype(result_data[column_name].dtype):
                    # For numeric columns, just increment the max value
                    numeric_values = [v for v in existing_values if isinstance(v, (int, float))]
                    max_val = max(numeric_values, default=0)
                    new_value = int(max_val) + 1
                else:
                    # For string columns, add a suffix to make it unique
                    prefix = str(current_value)
                    suffix_len = 4  # Use a smaller suffix than for PKs
                    
                    # Truncate prefix if needed to respect max_length
                    if max_length and len(prefix) + suffix_len > max_length:
                        prefix = prefix[:max_length - suffix_len]
                        
                    # Try to create a unique value
                    tries = 0
                    while True:
                        suffix = ''.join(random.choices(string.digits, k=suffix_len))
                        new_value = f"{prefix}{suffix}"
                        
                        # Truncate if needed
                        if max_length and len(new_value) > max_length:
                            new_value = new_value[:max_length]
                            
                        # Check if it's unique
                        if new_value not in existing_values:
                            break
                            
                        tries += 1
                        if tries > 1000:
                            # If we're struggling, use a longer suffix
                            suffix_len += 1
                            tries = 0
                            
                        if suffix_len > 10:
                            # Give up and log error
                            logger.warning(f"Could not generate unique value for {table_name}.{column_name}")
                            break
                
                # Update the value in the dataframe and add to our set
                result_data.at[idx, column_name] = new_value
                
            # Add the value to our set of known values
            existing_values.add(result_data.at[idx, column_name])
            
        # Update the global set of unique values
        self.all_unique_values[(table_name, column_name)] = existing_values
        
        return result_data
        
    def _enforce_string_length(self, data: pd.DataFrame, column_name: str, max_length: int) -> pd.DataFrame:
        """
        Enforce maximum string length for a column in the DataFrame.
        
        Args:
            data: DataFrame to process
            column_name: Name of the column to enforce length constraints
            max_length: Maximum allowed length for strings
            
        Returns:
            DataFrame with string length constraints enforced
        """
        if column_name not in data.columns:
            return data
            
        # Make a copy to avoid modifying the original
        result_data = data.copy()
        
        # Only process string values
        string_mask = result_data[column_name].apply(lambda x: isinstance(x, str))
        
        # Check which strings exceed the maximum length
        length_mask = result_data.loc[string_mask, column_name].str.len() > max_length
        
        # Combine masks to get only string values that exceed max_length
        combined_mask = pd.Series(False, index=result_data.index)
        combined_mask[string_mask.index[string_mask]] = length_mask
        
        # Truncate strings that are too long
        if combined_mask.any():
            # Count how many values were truncated
            truncated_count = combined_mask.sum()
            if truncated_count > 0:
                logger.debug(f"Truncated {truncated_count} values in column {column_name} to max length {max_length}")
                
            # Perform truncation
            result_data.loc[combined_mask, column_name] = result_data.loc[combined_mask, column_name].str.slice(0, max_length)
            
        return result_data

    def save_to_excel(self, output_path: str):
        """Save all generated data to an Excel file with one sheet per table."""
        logger.info(f"Saving generated data to {output_path}")
        
        # First save as pickle to handle any illegal Excel characters
        pickle_path = output_path.replace('.xlsx', '.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.generated_data, f)
        
        logger.info(f"Successfully saved {len(self.generated_data)} tables to {pickle_path}")
        logger.info(f"To convert to Excel, use: python export_data.py {pickle_path} {output_path}")

    def _enforce_foreign_keys(self, table_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure foreign key values reference only generated parent keys, supporting 
        composite keys and descriptive mappings from fk_mappings.xlsx.
        
        Two scenarios:
        1. If ChildKey is empty: Direct FK mapping (ChildColumn gets values from ParentColumn)
        2. If ChildKey is populated: Indirect FK mapping (ChildColumn gets values from ParentColumn 
           based on matching ChildKey with ParentKey)
        
        Args:
            table_name: Name of the table being processed
            data: DataFrame containing the generated data
            
        Returns:
            DataFrame with foreign key constraints enforced
        """
        result_data = data.copy()
        
        # Get foreign keys from database schema - these are always direct mappings
        fk_df = self.foreign_keys[table_name]
        for _, row in fk_df.iterrows():
            result_data = self._enforce_direct_fk(
                result_data, 
                row['parent_table'], 
                row['parent_column'], 
                row['child_column']
            )
            
        # Process additional FK mappings from fk_mappings.xlsx
        additional_fks = self.fk_mappings[
            (self.fk_mappings['ChildTable'] == table_name)
        ]
        
        for _, row in additional_fks.iterrows():
            parent_table = row['ParentTable']
            parent_column = row['ParentColumn']
            child_column = row['ChildColumn']
            parent_key = row.get('ParentKey')
            child_key = row.get('ChildKey')
            
            # Handle NaN values (pandas reads empty cells as NaN)
            if pd.isna(parent_key):
                parent_key = None
            if pd.isna(child_key):
                child_key = None
                
            if child_key is None or child_key == '':
                # Scenario 1: Direct FK mapping (ChildKey is empty)
                logger.info(f"Applying direct FK mapping: {table_name}.{child_column} -> {parent_table}.{parent_column}")
                result_data = self._enforce_direct_fk(
                    result_data, 
                    parent_table, 
                    parent_column, 
                    child_column
                )
            else:
                # Scenario 2: Indirect FK mapping (ChildKey is populated)
                logger.info(f"Applying indirect FK mapping: {table_name}.{child_column} -> {parent_table}.{parent_column} based on {child_key} -> {parent_key}")
                result_data = self._enforce_indirect_fk(
                    result_data,
                    parent_table,
                    parent_column,
                    parent_key,
                    child_column,
                    child_key
                )
            
        return result_data
    
    def _enforce_direct_fk(self, data: pd.DataFrame, parent_table: str, parent_column: str, child_column: str) -> pd.DataFrame:
        """
        Enforce direct foreign key mapping: ChildColumn gets values directly from ParentColumn.
        
        Args:
            data: DataFrame to update
            parent_table: Name of the parent table
            parent_column: Column in parent table to get values from
            child_column: Column in child table to update
            
        Returns:
            Updated DataFrame
        """
        if parent_table not in self.generated_data:
            logger.warning(f"Parent table {parent_table} not yet generated, skipping FK enforcement")
            return data
            
        parent_df = self.generated_data[parent_table]
        
        if parent_column not in parent_df.columns:
            logger.warning(f"Parent column {parent_column} not found in table {parent_table}")
            return data
            
        if child_column not in data.columns:
            logger.warning(f"Child column {child_column} not found in child table")
            return data
            
        # Get all valid parent values (remove NaN and duplicates)
        valid_parent_values = parent_df[parent_column].dropna().unique().tolist()
        
        if not valid_parent_values:
            logger.warning(f"No valid values found in {parent_table}.{parent_column}")
            return data
            
        result_data = data.copy()
        
        # Update each row in the child table
        for idx in result_data.index:
            # Randomly select a valid parent value
            chosen_value = random.choice(valid_parent_values)
            result_data.at[idx, child_column] = chosen_value
            
        logger.info(f"Updated {len(result_data)} rows in column {child_column} with values from {parent_table}.{parent_column}")
        return result_data
        
    def _enforce_indirect_fk(self, data: pd.DataFrame, parent_table: str, parent_column: str, 
                            parent_key: str, child_column: str, child_key: str) -> pd.DataFrame:
        """
        Enforce indirect foreign key mapping: ChildColumn gets values from ParentColumn 
        based on matching ChildKey with ParentKey.
        
        Args:
            data: DataFrame to update
            parent_table: Name of the parent table
            parent_column: Column in parent table to get values from
            parent_key: Key column in parent table for matching
            child_column: Column in child table to update
            child_key: Key column in child table for matching
            
        Returns:
            Updated DataFrame
        """
        if parent_table not in self.generated_data:
            logger.warning(f"Parent table {parent_table} not yet generated, skipping FK enforcement")
            return data
            
        parent_df = self.generated_data[parent_table]
        
        # Validate columns exist
        for col, table in [(parent_column, parent_table), (parent_key, parent_table)]:
            if col not in parent_df.columns:
                logger.warning(f"Column {col} not found in table {table}")
                return data
                
        for col in [child_column, child_key]:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in child table")
                return data
        
        # Create lookup dictionary: parent_key -> parent_column value
        parent_lookup = {}
        for _, row in parent_df.iterrows():
            key_val = row[parent_key]
            col_val = row[parent_column]
            if pd.notna(key_val) and pd.notna(col_val):
                parent_lookup[key_val] = col_val
                
        if not parent_lookup:
            logger.warning(f"No valid lookup values found in {parent_table}")
            return data
            
        result_data = data.copy()
        updated_count = 0
        
        # Update child column based on child key matches
        for idx in result_data.index:
            child_key_value = result_data.at[idx, child_key]
            
            if pd.notna(child_key_value) and child_key_value in parent_lookup:
                result_data.at[idx, child_column] = parent_lookup[child_key_value]
                updated_count += 1
            else:
                # If no match found, assign a random valid value from parent
                if parent_lookup:
                    result_data.at[idx, child_column] = random.choice(list(parent_lookup.values()))
                    updated_count += 1
                    
        logger.info(f"Updated {updated_count} rows in column {child_column} based on {child_key} -> {parent_key} mapping")
        return result_data
        

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate realistic mock data for SQL Server.")
    parser.add_argument("--tables", required=True, help="Path to target_tables.xlsx")
    parser.add_argument("--fks", required=True, help="Path to fk_mappings.xlsx")
    parser.add_argument("--uniques", required=True, help="Path to unique_constraints.xlsx")
    parser.add_argument("--hierarchy", required=True, help="Path to table_hierarchy.xlsx")
    parser.add_argument("--output", default="output/generated_data.xlsx", help="Output Excel file path")
    
    return parser.parse_args()

def main():
    """Main entry point for the data generation script."""
    args = parse_args()
    
    # Get the config directory from the input files
    config_dir = os.path.dirname(args.tables)
    
    # Setup output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize data generator
    generator = DataGenerator(config_dir)
    
    # Load schemas and sample data
    generator.load_schemas()
    
    # Generate data for all tables
    generator.generate_all_data()
    
    # Save to Excel
    generator.save_to_excel(args.output)
    
    logger.info("Mock data generation complete!")

if __name__ == "__main__":
    main()
