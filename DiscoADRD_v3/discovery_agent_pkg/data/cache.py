"""
Data Cache module for the Discovery Agent.

This module provides a centralized data cache that loads CSV and JSON data once
and reuses it throughout the agent's lifecycle, avoiding repeated disk I/O operations.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from ..ui.interface import ui


class DataCache:
    """Centralized data cache for CSV and JSON data."""
    
    def __init__(self):
        """Initialize the data cache."""
        self._original_csv_data: Optional[pd.DataFrame] = None
        self._json_dictionary: Optional[Dict[str, Any]] = None
        self._processed_csv_data: Optional[pd.DataFrame] = None
        self._column_descriptions: Optional[Dict[str, str]] = None
        self._csv_path: Optional[str] = None
        self._json_path: Optional[str] = None
        self._dataset_statistics: Optional[Dict[str, Any]] = None
        self._is_initialized = False
    
    def initialize(self, dataset_folder: str) -> bool:
        """
        Initialize the data cache by loading CSV and JSON files from the dataset folder.
        
        Args:
            dataset_folder (str): Path to folder containing CSV and JSON files
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            ui.print_info(f"🔍 Initializing data cache from folder: {dataset_folder}")
            
            # Auto-detect CSV and JSON files
            csv_path, json_path = self._detect_dataset_files(dataset_folder)
            
            # Load original CSV data
            ui.print_info(f"📊 Loading original CSV data from: {csv_path}")
            self._original_csv_data = pd.read_csv(csv_path)
            self._csv_path = csv_path
            
            # Load JSON dictionary
            ui.print_info(f"📚 Loading JSON dictionary from: {json_path}")
            with open(json_path, 'r') as f:
                self._json_dictionary = json.load(f)
            self._json_path = json_path
            
            # Pre-compute column descriptions
            ui.print_info(f"📝 Pre-computing column descriptions...")
            self._column_descriptions = self._compute_column_descriptions()
            
            # Pre-compute dataset statistics
            ui.print_info(f"📊 Pre-computing dataset statistics...")
            self._dataset_statistics = self._compute_dataset_statistics()
            
            self._is_initialized = True
            
            ui.print_success(f"✅ Data cache initialized successfully!")
            ui.print_info(f"   📊 Original data shape: {self._original_csv_data.shape[0]} rows, {self._original_csv_data.shape[1]} columns")
            ui.print_info(f"   📚 JSON dictionary: {len(self._json_dictionary)} variables")
            ui.print_info(f"   📝 Column descriptions: {len(self._column_descriptions)} descriptions")
            
            return True
            
        except Exception as e:
            ui.print_error(f"❌ Failed to initialize data cache: {str(e)}")
            return False
    
    def _detect_dataset_files(self, folder_path: str) -> Tuple[str, str]:
        """
        Auto-detect CSV and JSON files in the given folder.
        
        Args:
            folder_path (str): Path to folder containing dataset files
        
        Returns:
            tuple: (csv_path, json_path) - paths to the detected files
        
        Raises:
            ValueError: If required files are not found
        """
        # Get all files in the folder
        files = os.listdir(folder_path)
        
        # Find CSV files
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        json_files = [f for f in files if f.lower().endswith('.json')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {folder_path}")
        
        if not json_files:
            raise ValueError(f"No JSON files found in folder: {folder_path}")
        
        # Use the first CSV and JSON files found
        csv_file = csv_files[0]
        json_file = json_files[0]
        
        csv_path = os.path.join(folder_path, csv_file)
        json_path = os.path.join(folder_path, json_file)
        
        ui.print_info(f"🔍 Auto-detected files:")
        ui.print_info(f"   📊 CSV: {csv_file}")
        ui.print_info(f"   📚 JSON: {json_file}")
        
        # Validate JSON file format
        try:
            with open(json_path, 'r') as f:
                json.load(f)
            ui.print_success(f"✅ JSON dictionary file is valid: {json_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_file}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file {json_file}: {str(e)}")
        
        return csv_path, json_path
    
    def _compute_column_descriptions(self) -> Dict[str, str]:
        """
        Pre-compute column descriptions for all columns in the original dataset.
        
        Returns:
            Dict[str, str]: Dictionary mapping column names to their descriptions
        """
        column_descriptions = {}
        
        for col in self._original_csv_data.columns:
            description = "No description available"
            
            # Try to find exact match in dictionary
            for var_name, var_info in self._json_dictionary.items():
                if col.upper() == var_name.upper():
                    description = var_info.get('Description', 'No description available')
                    break
            
            column_descriptions[col] = description
        
        return column_descriptions
    
    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        """
        Pre-compute comprehensive statistics about the original dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics
        """
        try:
            df = self._original_csv_data
            
            # Basic dataset info
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_data = df.isnull().sum().sum()
            missing_percentage = (missing_data / (total_rows * total_cols)) * 100
            
            # Data type counts
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Comprehensive statistics for numeric columns
            numeric_stats = {}
            if len(numeric_cols) > 0:
                numeric_df = df[numeric_cols]
                numeric_stats = {
                    'count': len(numeric_cols),
                    'column_names': numeric_cols,
                    'summary_stats': {}
                }
                
                # Get comprehensive stats for all numeric columns
                for col in numeric_cols:
                    if not numeric_df[col].isnull().all():
                        col_stats = numeric_df[col].describe()
                        missing_pct = round((numeric_df[col].isnull().sum() / total_rows) * 100, 1)
                        numeric_stats['summary_stats'][col] = {
                            'mean': round(col_stats['mean'], 2) if not pd.isna(col_stats['mean']) else 'N/A',
                            'std': round(col_stats['std'], 2) if not pd.isna(col_stats['std']) else 'N/A',
                            'min': round(col_stats['min'], 2) if not pd.isna(col_stats['min']) else 'N/A',
                            'max': round(col_stats['max'], 2) if not pd.isna(col_stats['max']) else 'N/A',
                            'missing_pct': missing_pct
                        }
            
            # Comprehensive info for categorical columns
            categorical_stats = {}
            if len(categorical_cols) > 0:
                categorical_stats = {
                    'count': len(categorical_cols),
                    'column_names': categorical_cols,
                    'summary_info': {}
                }
                
                # Get comprehensive info for all categorical columns
                for col in categorical_cols:
                    if not df[col].isnull().all():
                        unique_vals = df[col].nunique()
                        most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                        missing_pct = round((df[col].isnull().sum() / total_rows) * 100, 1)
                        categorical_stats['summary_info'][col] = {
                            'unique_values': unique_vals,
                            'most_common': str(most_common),
                            'missing_pct': missing_pct
                        }
            
            return {
                'total_rows': total_rows,
                'total_cols': total_cols,
                'missing_data': missing_data,
                'missing_percentage': round(missing_percentage, 1),
                'numeric_cols': len(numeric_cols),
                'categorical_cols': len(categorical_cols),
                'all_columns': df.columns.tolist(),
                'numeric_column_names': numeric_cols,
                'categorical_column_names': categorical_cols,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats
            }
            
        except Exception as e:
            return {
                'error': f"Could not compute dataset statistics: {str(e)}",
                'total_rows': 0,
                'total_cols': 0,
                'missing_data': 0,
                'missing_percentage': 0,
                'numeric_cols': 0,
                'categorical_cols': 0,
                'all_columns': [],
                'numeric_column_names': [],
                'categorical_column_names': [],
                'numeric_stats': {},
                'categorical_stats': {}
            }
    
    # Getters for cached data
    def get_original_csv_data(self) -> pd.DataFrame:
        """Get the original CSV data."""
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        return self._original_csv_data.copy()
    
    def get_json_dictionary(self) -> Dict[str, Any]:
        """Get the JSON dictionary."""
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        return self._json_dictionary.copy()
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get pre-computed column descriptions."""
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        return self._column_descriptions.copy()
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get pre-computed dataset statistics."""
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        return self._dataset_statistics.copy()
    
    def get_csv_path(self) -> str:
        """Get the CSV file path."""
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        return self._csv_path
    
    def get_json_path(self) -> str:
        """Get the JSON file path."""
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        return self._json_path
    
    def is_initialized(self) -> bool:
        """Check if the data cache is initialized."""
        return self._is_initialized
    
    # Methods for processed data management
    def set_processed_data(self, processed_df: pd.DataFrame) -> None:
        """
        Store processed data in the cache.
        
        Args:
            processed_df (pd.DataFrame): The processed dataframe to store
        """
        self._processed_csv_data = processed_df.copy()
        ui.print_info(f"💾 Stored processed data in cache: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get the processed data if available."""
        if self._processed_csv_data is not None:
            return self._processed_csv_data.copy()
        return None
    
    def has_processed_data(self) -> bool:
        """Check if processed data is available."""
        return self._processed_csv_data is not None
    
    def clear_processed_data(self) -> None:
        """Clear the processed data from cache."""
        self._processed_csv_data = None
        ui.print_info("🧹 Cleared processed data from cache")
    
    def get_processed_column_descriptions(self) -> Dict[str, str]:
        """
        Get column descriptions for the processed dataset.
        This handles both original columns and dummy variables.
        
        Returns:
            Dict[str, str]: Dictionary mapping column names to their descriptions
        """
        if not self._is_initialized:
            raise RuntimeError("Data cache not initialized. Call initialize() first.")
        
        if not self.has_processed_data():
            raise RuntimeError("No processed data available. Call set_processed_data() first.")
        
        import re
        
        processed_descriptions = {}
        
        for col in self._processed_csv_data.columns:
            description = "No description available"
            
            # Check if it's a dummy variable (contains underscore)
            if '_' in col:
                # Try to extract the original variable name from dummy variable
                possible_base_vars = []
                for var_name in self._json_dictionary.keys():
                    if col.startswith(var_name + '_'):
                        possible_base_vars.append(var_name)
                
                if possible_base_vars:
                    # Use the longest matching base variable name
                    base_var = max(possible_base_vars, key=len)
                    var_info = self._json_dictionary[base_var]
                    base_description = var_info.get('Description', 'No description available')
                    
                    # Extract the category part
                    category_part = col.replace(f"{base_var}_", "")
                    
                    # Try to get the category description from the values
                    if 'Values' in var_info:
                        # The category_part is now cleaned (uppercase, spaces replaced with underscores), so we need to find the original
                        category_desc = None
                        for value_code, value_desc in var_info['Values'].items():
                            # Clean the value description the same way we cleaned the category names
                            cleaned_desc = re.sub(r'\s+', '_', str(value_desc))  # Replace spaces with underscores
                            cleaned_desc = re.sub(r'[^\w_]', '', cleaned_desc)  # Remove special characters except underscores
                            cleaned_desc = cleaned_desc.upper()  # Make uppercase
                            if cleaned_desc == category_part:
                                category_desc = value_desc
                                break
                        
                        if category_desc:
                            description = f"{base_description} - Category: {category_desc}"
                        else:
                            description = f"{base_description} - Category: {category_part}"
                    else:
                        description = f"{base_description} - Category: {category_part}"
                else:
                    # If we can't find the base variable, try direct match
                    for var_name, var_info in self._json_dictionary.items():
                        if col.upper() == var_name.upper():
                            description = var_info.get('Description', 'No description available')
                            break
            else:
                # Regular column - try to find exact match
                for var_name, var_info in self._json_dictionary.items():
                    if col.upper() == var_name.upper():
                        description = var_info.get('Description', 'No description available')
                        break
            
            processed_descriptions[col] = description
        
        return processed_descriptions
    
    def get_processed_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the processed dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing processed dataset statistics
        """
        if not self.has_processed_data():
            raise RuntimeError("No processed data available. Call set_processed_data() first.")
        
        try:
            df = self._processed_csv_data
            
            # Basic dataset info
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_data = df.isnull().sum().sum()
            missing_percentage = (missing_data / (total_rows * total_cols)) * 100
            
            # Data type counts
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Comprehensive statistics for numeric columns
            numeric_stats = {}
            if len(numeric_cols) > 0:
                numeric_df = df[numeric_cols]
                numeric_stats = {
                    'count': len(numeric_cols),
                    'column_names': numeric_cols,
                    'summary_stats': {}
                }
                
                # Get comprehensive stats for all numeric columns
                for col in numeric_cols:
                    if not numeric_df[col].isnull().all():
                        col_stats = numeric_df[col].describe()
                        missing_pct = round((numeric_df[col].isnull().sum() / total_rows) * 100, 1)
                        numeric_stats['summary_stats'][col] = {
                            'mean': round(col_stats['mean'], 2) if not pd.isna(col_stats['mean']) else 'N/A',
                            'std': round(col_stats['std'], 2) if not pd.isna(col_stats['std']) else 'N/A',
                            'min': round(col_stats['min'], 2) if not pd.isna(col_stats['min']) else 'N/A',
                            'max': round(col_stats['max'], 2) if not pd.isna(col_stats['max']) else 'N/A',
                            'missing_pct': missing_pct
                        }
            
            # Comprehensive info for categorical columns
            categorical_stats = {}
            if len(categorical_cols) > 0:
                categorical_stats = {
                    'count': len(categorical_cols),
                    'column_names': categorical_cols,
                    'summary_info': {}
                }
                
                # Get comprehensive info for all categorical columns
                for col in categorical_cols:
                    if not df[col].isnull().all():
                        unique_vals = df[col].nunique()
                        most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                        missing_pct = round((df[col].isnull().sum() / total_rows) * 100, 1)
                        categorical_stats['summary_info'][col] = {
                            'unique_values': unique_vals,
                            'most_common': str(most_common),
                            'missing_pct': missing_pct
                        }
            
            return {
                'total_rows': total_rows,
                'total_cols': total_cols,
                'missing_data': missing_data,
                'missing_percentage': round(missing_percentage, 1),
                'numeric_cols': len(numeric_cols),
                'categorical_cols': len(categorical_cols),
                'all_columns': df.columns.tolist(),
                'numeric_column_names': numeric_cols,
                'categorical_column_names': categorical_cols,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats
            }
            
        except Exception as e:
            return {
                'error': f"Could not compute processed dataset statistics: {str(e)}",
                'total_rows': 0,
                'total_cols': 0,
                'missing_data': 0,
                'missing_percentage': 0,
                'numeric_cols': 0,
                'categorical_cols': 0,
                'all_columns': [],
                'numeric_column_names': [],
                'categorical_column_names': [],
                'numeric_stats': {},
                'categorical_stats': {}
            }
    
    def sample_dataset_rows(self, sample_size: int, seed: int = None) -> pd.DataFrame:
        """
        Sample a subset of rows from the original dataset.
        
        Args:
            sample_size (int): Number of rows to sample
            seed (int): Random seed for reproducible sampling
        
        Returns:
            pd.DataFrame: Sampled dataset
        """
        if not self._is_initialized or self._original_csv_data is None:
            raise ValueError("Data cache not initialized or no original data available")
        
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")
        
        df = self._original_csv_data
        total_rows = len(df)
        
        if sample_size >= total_rows:
            ui.print_info(f"📊 Sample size ({sample_size}) >= total rows ({total_rows}), using full dataset")
            return df.copy()
        
        # Sample rows
        sampled_df = df.sample(n=sample_size, random_state=seed).copy()
        
        ui.print_info(f"📊 Sampled {len(sampled_df)} rows from {total_rows} total rows")
        
        return sampled_df