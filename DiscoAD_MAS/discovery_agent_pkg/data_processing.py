"""
Data processing module for the Discovery Agent.

This module handles all data-related operations including:
- Dataset statistics and analysis
- CSV filtering and preprocessing
- Variable selection and relevance analysis
- File detection and validation
"""

import os
import json
import re
from typing import Dict, Any, List, Tuple
from .ui import ui
from .models import ModelClient

# Optional pandas and numpy imports with fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create dummy classes for when pandas is not available
    class pd:
        @staticmethod
        def read_csv(path):
            raise ImportError("pandas is required for CSV operations. Please install pandas.")
        
        @staticmethod
        def isna(value):
            return value is None or str(value).lower() in ['nan', 'none', '']
    
    class np:
        @staticmethod
        def isnan(value):
            try:
                return float(value) != float(value)  # NaN check
            except (ValueError, TypeError):
                return False


class DataProcessor:
    """Handles all data processing operations for the Discovery Agent."""
    
    def __init__(self, model_client: ModelClient):
        """Initialize the data processor with a model client."""
        self.model_client = model_client
    
    def get_dataset_statistics(self, csv_path: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset from the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics
        """
        if not PANDAS_AVAILABLE:
            return {
                'error': "pandas is required for dataset statistics. Please install pandas.",
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
        
        try:
            df = pd.read_csv(csv_path)
            
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
                            'most_common': str(most_common),  # No truncation
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
                'error': f"Could not load CSV for statistics: {str(e)}",
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
    
    def get_relevant_variables(self, hypothesis: str, json_path: str, csv_statistics: Dict[str, Any] = None, previous_iteration_context: str = None, previously_used_variables: List[str] = None) -> Dict[str, str]:
        """
        Get all relevant variables for a hypothesis from a JSON dictionary.
        
        This method uses an LLM to analyze a hypothesis and identify all relevant
        variables from a data dictionary. It returns a simple dictionary with
        variable names as keys and their descriptions as values.
        
        Args:
            hypothesis (str): The hypothesis to analyze
            json_path (str): Path to the JSON dictionary file
            csv_statistics (Dict[str, Any], optional): Statistics from the CSV file
            previous_iteration_context (str, optional): Context from previous outer loop iteration
            previously_used_variables (List[str], optional): List of variables used in previous iterations
        
        Returns:
            Dict[str, str]: Dictionary with variable names as keys and descriptions as values
        """
        
        # Load the JSON dictionary
        with open(json_path, 'r') as f:
            dictionary = json.load(f)
        
        # Format CSV statistics if provided
        csv_stats = ""
        if csv_statistics:
            if 'error' in csv_statistics:
                csv_stats = f"\n\nNote: {csv_statistics['error']}\n"
            else:
                # Format comprehensive dataset overview
                csv_stats = (
                    f"\n\nDATASET OVERVIEW:\n"
                    f"- Total rows: {csv_statistics['total_rows']:,}\n"
                    f"- Total columns: {csv_statistics['total_cols']}\n"
                    f"- Missing values: {csv_statistics['missing_data']:,} ({csv_statistics['missing_percentage']}%)\n"
                    f"- Numeric columns: {csv_statistics['numeric_cols']}\n"
                    f"- Categorical columns: {csv_statistics['categorical_cols']}\n"
                )
                
                # Add all column names for reference
                if csv_statistics.get('all_columns'):
                    csv_stats += f"- All available columns: {', '.join(csv_statistics['all_columns'])}\n"
                
                # Add numeric column names if available
                if csv_statistics.get('numeric_column_names'):
                    csv_stats += f"- Numeric columns: {', '.join(csv_statistics['numeric_column_names'])}\n"
                
                # Add categorical column names if available
                if csv_statistics.get('categorical_column_names'):
                    csv_stats += f"- Categorical columns: {', '.join(csv_statistics['categorical_column_names'])}\n"
        else:
            csv_stats = "\n\nNote: No CSV statistics provided.\n"
        
        # Create a formatted string of all keys and descriptions
        keys_info = []
        for key, info in dictionary.items():
            description = info.get('Description', 'No description available')
            keys_info.append(f"Key: {key} | Description: {description}")
        
        keys_text = "\n".join(keys_info)
        
        # Create the prompt for the LLM
        context_section = ""
        if previous_iteration_context:
            context_section = (
                f"\n\nPREVIOUS ITERATION CONTEXT:\n"
                f"{previous_iteration_context}\n\n"
            )
        
        # Add information about previously used variables
        previously_used_section = ""
        if previously_used_variables:
            previously_used_section = (
                f"\n\nPREVIOUSLY USED VARIABLES:\n"
                f"The following variables were used in previous iterations: {', '.join(previously_used_variables)}\n"
                f"Consider selecting DIFFERENT variables this time to explore new aspects of the hypothesis. "
                f"Only reuse previously used variables if they are absolutely essential for this specific analysis.\n\n"
            )
        
        prompt = (
            f"You are a data analysis expert. Given a hypothesis and a dictionary of data fields, identify the most useful keys for testing or analyzing this hypothesis. "
            "Focus on selecting variables that would enhance the scientific rigor and comprehensiveness of the analysis."
            f"{context_section}"
            f"{previously_used_section}"
            f'Hypothesis: "{hypothesis}"\n\n'
            f"Available data fields:\n"
            f"{keys_text}\n"
            f"{csv_stats}\n"
            f"DATASET OVERVIEW:\n"
            f"- Total variables in dictionary: {len(dictionary)}\n"
            f"- Variables will be filtered from the full dataset\n"
            f"- Sample size and data quality will be assessed after filtering\n"
            f"- Missing data patterns will be evaluated during analysis\n\n"
            "CRITICAL: Select as FEW variables as possible while still being able to test the hypothesis effectively. "
            "Aim for 10-20 variables maximum, not 50. Focus on the most essential variables only. "
            "More variables = more missing data issues and complexity. Fewer variables = cleaner analysis and better results.\n\n"
            "SELECTION STRATEGY:\n"
            "1. Identify the core variables directly related to the hypothesis\n"
            "2. Add only essential demographic/control variables (age, sex, education)\n"
            "3. Include only the most relevant medical/clinical variables\n"
            "4. Avoid redundant or highly correlated variables\n"
            "5. Prioritize variables with good data quality (low missingness)\n\n"
            "Please identify the most useful keys for this hypothesis. For each key, provide:\n"
            "1. The key name\n"
            "2. A brief explanation of why it would be valuable for this analysis\n\n"
            "Format your response as a numbered list with the key name and explanation.\n\n"
            "At the end of your response, provide a Python list of the variable names you identified (aim for 10-20, maximum 30), formatted as:\n"
            'variable_names = ["key1", "key2", "key3", ...]'
        )

        # Generate response using the model client
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.model_client.generate(messages)
        print(f"Data Processing Response: {response}")
        
        # Parse the response to extract variable names using regex
        variable_names = []
        
        # First, try to extract from the Python list using regex
        python_list_pattern = r'variable_names\s*=\s*\[(.*?)\]'
        python_list_match = re.search(python_list_pattern, response, re.DOTALL)
        
        if python_list_match:
            try:
                # Extract the list content and parse it
                list_content = python_list_match.group(1)
                # Use regex to extract quoted strings from the list
                quoted_strings = re.findall(r'"([^"]*)"', list_content)
                if quoted_strings:
                    variable_names = quoted_strings
                    print(f"Extracted variable names from Python list using regex: {variable_names}")
                else:
                    # Try single quotes
                    single_quoted_strings = re.findall(r"'([^']*)'", list_content)
                    if single_quoted_strings:
                        variable_names = single_quoted_strings
                        print(f"Extracted variable names from Python list using regex (single quotes): {variable_names}")
            except Exception as e:
                print(f"Failed to parse Python list with regex: {e}")
        
        # If Python list parsing failed, return empty dictionary
        if not variable_names:
            print("No variable names found in Python list, returning empty dictionary")
            return {}
        
        # Create result dictionary with variable names and descriptions
        result_dict = {}
        for var_name in variable_names:
            if var_name in dictionary:
                result_dict[var_name] = dictionary[var_name].get('Description', 'No description available')
        
        return result_dict
    
    def preprocess_data(self, df: pd.DataFrame, missing_threshold: float = 0.10) -> pd.DataFrame:
        """
        Perform data preprocessing by removing columns with high missingness and handling remaining data.
        
        Args:
            df (pd.DataFrame): Input dataframe to preprocess
            missing_threshold (float): Threshold for missing data percentage (default 10%)
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data preprocessing operations. Please install pandas.")
        
        ui.print_info(f"🔧 Starting data preprocessing with missing data removal...")
        original_shape = df.shape
        ui.print_info(f"📊 Original data shape: {original_shape[0]} rows, {original_shape[1]} columns")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # 1. Analyze missing data patterns
        ui.print_info(f"🔍 Analyzing missing data patterns...")
        missing_analysis = self._analyze_missing_data(processed_df)
        self._print_missing_analysis(missing_analysis)
        
        # 2. Remove columns with high missingness (>threshold)
        ui.print_info(f"🗑️ Removing columns with >{missing_threshold*100:.1f}% missing data...")
        columns_to_remove = missing_analysis['high_missing_cols']
        
        if columns_to_remove:
            ui.print_info(f"   📊 Removing {len(columns_to_remove)} columns with high missingness:")
            for col in columns_to_remove:
                missing_pct = missing_analysis['missing_percentages'][col]
                ui.print_info(f"   • {col}: {missing_pct:.1f}% missing")
            processed_df = processed_df.drop(columns=columns_to_remove)
        else:
            ui.print_info(f"   ✅ No columns exceed the {missing_threshold*100:.1f}% missing threshold")
        
        # 3. Handle remaining missing values in kept columns
        remaining_missing = processed_df.isnull().sum().sum()
        if remaining_missing > 0:
            ui.print_info(f"🔧 Handling remaining {remaining_missing} missing values...")
            processed_df = self._handle_remaining_missing(processed_df)
        
        # 4. Create dummy variables for categorical columns
        ui.print_info(f"🔧 Creating dummy variables for categorical columns...")
        processed_df = self._create_dummy_variables(processed_df)
        
        # 5. Convert remaining object columns to numeric if possible
        ui.print_info(f"🔧 Converting remaining object columns to numeric...")
        processed_df = self._convert_to_numeric(processed_df)
        
        # 6. Final data validation
        ui.print_info(f"🔍 Final data validation...")
        final_missing = processed_df.isnull().sum().sum()
        if final_missing > 0:
            ui.print_warning(f"⚠️ Still {final_missing} missing values after preprocessing, filling with 0")
            processed_df.fillna(0, inplace=True)
        
        final_shape = processed_df.shape
        ui.print_success(f"✅ Data preprocessing completed!")
        ui.print_info(f"📊 Final data shape: {final_shape[0]} rows, {final_shape[1]} columns")
        ui.print_info(f"📈 Columns removed: {original_shape[1] - final_shape[1]}")
        ui.print_info(f"📈 Columns added: {final_shape[1] - (original_shape[1] - len(columns_to_remove))}")
        
        return processed_df
    
    def _analyze_missing_data(self, df: pd.DataFrame, missing_threshold: float = 0.10) -> Dict[str, Any]:
        """
        Analyze missing data patterns in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            missing_threshold (float): Threshold for high missingness
        
        Returns:
            Dict[str, Any]: Missing data analysis results
        """
        total_rows = len(df)
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / total_rows * 100).round(2)
        
        # Identify columns with high missingness
        high_missing_cols = missing_percentages[missing_percentages > missing_threshold * 100].index.tolist()
        
        # Summary statistics
        total_missing = missing_counts.sum()
        total_cells = total_rows * len(df.columns)
        overall_missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        return {
            'total_rows': total_rows,
            'total_columns': len(df.columns),
            'total_missing': total_missing,
            'overall_missing_pct': round(overall_missing_pct, 2),
            'missing_counts': missing_counts,
            'missing_percentages': missing_percentages,
            'high_missing_cols': high_missing_cols,
            'missing_threshold': missing_threshold
        }
    
    def _print_missing_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print missing data analysis results."""
        ui.print_info(f"   📊 Total missing values: {analysis['total_missing']:,} ({analysis['overall_missing_pct']:.1f}% of all data)")
        ui.print_info(f"   📊 Columns with high missingness (>10%): {len(analysis['high_missing_cols'])}")
        
        if analysis['high_missing_cols']:
            ui.print_info(f"   📊 Top 10 columns with highest missingness:")
            top_missing = analysis['missing_percentages'].nlargest(10)
            for col, pct in top_missing.items():
                ui.print_info(f"     • {col}: {pct:.1f}% missing")
    
    def _handle_remaining_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle remaining missing values in columns that were kept.
        
        Args:
            df (pd.DataFrame): Dataframe with remaining missing values
        
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        processed_df = df.copy()
        
        # For numeric columns: impute with median (more robust than mean)
        numeric_cols = processed_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if processed_df[col].isnull().sum() > 0:
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
                ui.print_info(f"   📊 Imputed {processed_df[col].isnull().sum()} missing values in {col} with median: {median_val:.2f}")
        
        # For categorical columns: impute with mode
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if processed_df[col].isnull().sum() > 0:
                mode_val = processed_df[col].mode()
                if len(mode_val) > 0:
                    processed_df[col].fillna(mode_val[0], inplace=True)
                    ui.print_info(f"   📊 Imputed {processed_df[col].isnull().sum()} missing values in {col} with mode: {mode_val[0]}")
                else:
                    # If no mode (all values are missing), fill with 'Unknown'
                    processed_df[col].fillna('Unknown', inplace=True)
                    ui.print_info(f"   📊 Imputed {processed_df[col].isnull().sum()} missing values in {col} with 'Unknown'")
        
        return processed_df
    
    def _create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dummy variables for categorical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with dummy variables
        """
        processed_df = df.copy()
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Check if column has reasonable number of unique values
            unique_count = processed_df[col].nunique()
            if unique_count > 1 and unique_count <= 20:  # Reasonable threshold for dummy variables
                # Create dummy variables
                dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                # Add dummy variables to dataframe
                processed_df = pd.concat([processed_df, dummies], axis=1)
                # Remove original categorical column
                processed_df.drop(col, axis=1, inplace=True)
                ui.print_info(f"   📊 Created {len(dummies.columns)} dummy variables for {col}")
            elif unique_count > 20:
                ui.print_warning(f"   ⚠️ Column {col} has {unique_count} unique values, skipping dummy variable creation")
            else:
                ui.print_info(f"   📊 Column {col} has only 1 unique value, removing it")
                processed_df.drop(col, axis=1, inplace=True)
        
        return processed_df
    
    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert remaining object columns to numeric if possible.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with converted columns
        """
        processed_df = df.copy()
        
        for col in processed_df.select_dtypes(include=['object']).columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                ui.print_info(f"   📊 Converted {col} to numeric")
            except:
                ui.print_warning(f"   ⚠️ Could not convert {col} to numeric, keeping as object")
        
        return processed_df
    

    def filter_csv_with_dictionary(self, folder_path: str, relevant_variables: Dict[str, str], hypothesis: str) -> Tuple[str, Dict[str, str]]:
        """
        Filter CSV files in a folder using relevant variables from dictionary, preprocess the data, and save to .working.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            relevant_variables (Dict[str, str]): Dictionary of relevant variable names and descriptions
            hypothesis (str): The hypothesis being tested
        
        Returns:
            tuple[str, Dict[str, str]]: Path to the filtered CSV file and dictionary of variable datatypes
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV filtering operations. Please install pandas.")
        
        from pathlib import Path
        
        ui.print_info(f"🔍 Filtering CSV files with relevant variables...")
        
        # Get list of CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {folder_path}")
        
        # Use the first CSV file (you can modify this logic as needed)
        csv_file = csv_files[0]
        csv_path = os.path.join(folder_path, csv_file)
        
        ui.print_info(f"📊 Processing CSV file: {csv_file}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            ui.print_info(f"📈 Original CSV has {len(df)} rows and {len(df.columns)} columns")
            
            # Get the relevant column names (case-insensitive matching)
            relevant_columns = []
            available_columns = df.columns.tolist()
            
            for var_name in relevant_variables.keys():
                # Try exact match first
                if var_name in available_columns:
                    relevant_columns.append(var_name)
                else:
                    # Try case-insensitive match
                    for col in available_columns:
                        if col.upper() == var_name.upper():
                            relevant_columns.append(col)
                            break
            
            ui.print_info(f"🎯 Found {len(relevant_columns)} relevant columns: {relevant_columns}")
            
            if not relevant_columns:
                ui.print_warning(f"⚠️ No relevant columns found. Using all columns.")
                filtered_df = df
            else:
                # Filter to only relevant columns
                filtered_df = df[relevant_columns]
                
                # Rename columns to their descriptions from the dictionary
                column_mapping = {}
                for col in filtered_df.columns:
                    # Find the corresponding variable name and its description
                    for var_name, description in relevant_variables.items():
                        if col.upper() == var_name.upper():
                            # Use the description as the new column name
                            # Remove special characters and replace spaces with underscores for variable names
                            clean_description = re.sub(r"[^\w\s]", "", description)  # Remove special characters
                            clean_description = re.sub(r"\s+", "_", clean_description.strip())  # Replace spaces with underscores
                            column_mapping[col] = clean_description
                            break
                
                # Apply the column renaming
                if column_mapping:
                    filtered_df = filtered_df.rename(columns=column_mapping)
                    ui.print_info(f"🔄 Renamed columns to descriptions: {column_mapping}")
            
            # Preprocess the data (impute missing values, create dummy variables, etc.)
            processed_df = self.preprocess_data(filtered_df)
            
            # Create .working directory if it doesn't exist
            working_dir = Path("working")
            working_dir.mkdir(exist_ok=True)
            
            # Save processed CSV to .working directory
            output_filename = f"filtered_dataset.csv"
            output_path = working_dir / output_filename
            
            processed_df.to_csv(output_path, index=False)
            
            ui.print_success(f"✅ Processed CSV saved to: {output_path}")
            ui.print_info(f"📊 Final processed CSV has {len(processed_df)} rows and {len(processed_df.columns)} columns")
            
            # Get datatypes of the processed variables using column names as keys
            variable_datatypes = {}
            for col in processed_df.columns:
                dtype = str(processed_df[col].dtype)
                variable_datatypes[col] = dtype
            
            return str(output_path), variable_datatypes
            
        except Exception as e:
            error_msg = f"Error filtering and preprocessing CSV file: {str(e)}"
            ui.print_error(error_msg, "csv_filtering")
            raise RuntimeError(error_msg) from e
    
    def detect_dataset_files(self, folder_path: str) -> Tuple[str, str]:
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
    
    def run_manual_preprocessing_conversation(self, hypothesis: str, dataset_folder: str = None, previous_iteration_context: str = None, conversation_manager=None, previously_used_variables: List[str] = None):
        """
        Run manual preprocessing conversation where user provides folder path containing both CSV and JSON files.
        This method uses the provided dataset folder or asks the user for:
        1. Path to folder containing both CSV and JSON files
        
        Then it:
        1. Auto-detects CSV and JSON files in the folder
        2. Gets relevant variables from the JSON dictionary using the provided hypothesis
        3. Filters CSV files using the relevant variables
        4. Saves filtered data to .working directory
        
        Args:
            hypothesis (str): The hypothesis to test (provided as parameter)
            dataset_folder (str): Path to folder containing CSV and JSON files (from config)
            previous_iteration_context (str, optional): Context from previous outer loop iteration
            conversation_manager: Conversation manager to add system messages
            previously_used_variables (List[str], optional): List of variables used in previous iterations
        
        Returns:
            Dict[str, Any]: Result dictionary with success status and data
        """
        ui.print_info(f"🔧 Starting manual preprocessing conversation...")
        
        try:
            # Use provided dataset folder or get from user
            if dataset_folder:
                folder_path = dataset_folder
                ui.print_info(f"📁 Using dataset folder from config: {folder_path}")
            else:
                folder_path = ui.get_folder_path("Enter the path to the folder containing your dataset (CSV and JSON files)")
            
            # Auto-detect CSV and JSON files in the folder
            csv_path, json_path = self.detect_dataset_files(folder_path)
            
            if not hypothesis.strip():
                ui.print_warning(f"⚠️ No hypothesis provided. Exiting manual preprocessing.")
                return {'success': False, 'error': 'No hypothesis provided'}
            
            ui.print_info(f"🔬 Hypothesis: {hypothesis}")
            ui.print_info(f"📁 Dataset folder: {folder_path}")
            ui.print_info(f"📊 CSV file: {csv_path}")
            ui.print_info(f"📚 JSON dictionary: {json_path}")
            
            # Get CSV statistics
            csv_statistics = self.get_dataset_statistics(csv_path)
            
            # Get relevant variables from dictionary
            ui.print_info(f"🔍 Analyzing hypothesis to find relevant variables...")
            if previous_iteration_context:
                ui.print_info(f"🔄 Using previous iteration context to guide variable selection...")
            if previously_used_variables:
                ui.print_info(f"📋 Avoiding {len(previously_used_variables)} previously used variables: {', '.join(previously_used_variables[:5])}{'...' if len(previously_used_variables) > 5 else ''}")
            
            relevant_variables = self.get_relevant_variables(hypothesis, json_path, csv_statistics, previous_iteration_context, previously_used_variables)
            
            if not relevant_variables:
                ui.print_warning(f"⚠️ No relevant variables found. Using all available data.")
                # Create a dummy dictionary with all columns (we'll handle this in filtering)
                relevant_variables = {"ALL_COLUMNS": "All available columns"}
            
            ui.print_info(f"🎯 Found {len(relevant_variables)} relevant variables:")
            for var_name, description in relevant_variables.items():
                ui.print_info(f"   • {var_name}: {description}")

            # Filter CSV files using relevant variables
            filtered_csv_path, variable_datatypes = self.filter_csv_with_dictionary(folder_path, relevant_variables, hypothesis)
            
            # Get statistics for the filtered dataset
            filtered_csv_statistics = self.get_dataset_statistics(filtered_csv_path)
            
            variable_datatypes_str = "\n".join([f"- {col}: {dtype}" for col, dtype in variable_datatypes.items()])

            # Add system message to conversation if conversation_manager is provided
            if conversation_manager:
                summarized_messages = [{
                    "role": "System",
                    "content": (
                        f"Filtered data saved to: {filtered_csv_path}\n\n"
                        f"Relevant variables for this hypothesis:\n{variable_datatypes_str}\n\n"
                        f"Filtered dataset statistics:\n"
                        f"- Total rows: {filtered_csv_statistics.get('total_rows', 'N/A'):,}\n"
                        f"- Total columns: {filtered_csv_statistics.get('total_cols', 'N/A')}\n"
                        f"- Missing values: {filtered_csv_statistics.get('missing_data', 'N/A'):,} ({filtered_csv_statistics.get('missing_percentage', 'N/A')}%)\n"
                        f"- Numeric columns: {filtered_csv_statistics.get('numeric_cols', 'N/A')}\n"
                        f"- Categorical columns: {filtered_csv_statistics.get('categorical_cols', 'N/A')}"
                    )
                }]
                conversation_manager.messages.extend(summarized_messages)

            ui.print_info(f"✅ Manual preprocessing completed successfully!")
            ui.print_info(f"📊 Filtered data saved to: {filtered_csv_path}")
            
            # Return the information needed for enhanced planning
            return {
                'success': True,
                'hypothesis': hypothesis,
                'filtered_csv_statistics': filtered_csv_statistics,
                'relevant_variables': relevant_variables,
                'filtered_csv_path': filtered_csv_path
            }
            
        except Exception as e:
            error_msg = f"Manual preprocessing failed: {str(e)}"
            ui.print_error(error_msg, "manual_preprocessing")
            return {
                'success': False,
                'error': error_msg
            }
