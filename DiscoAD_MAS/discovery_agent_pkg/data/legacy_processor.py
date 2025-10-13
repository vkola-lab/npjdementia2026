"""
Legacy Data Processing for the Discovery Agent.

This module contains the original complex data processing approach.
Kept for backward compatibility and reference.
"""

import os
import json
import re
from typing import Dict, Any, List, Tuple
from ..ui.interface import ui
from ..agents.models import ModelClient

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

# =============================================================================
# VARIABLE CATEGORIZATION FUNCTIONS
# =============================================================================

def _categorize_single_variable(var_name: str, var_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Categorize a single variable based on its JSON dictionary information.
    
    Args:
        var_name (str): Variable name
        var_info (Dict[str, Any]): Variable information from JSON dictionary
    
    Returns:
        Dict[str, Any]: Categorized variable information
    """
    values_info = var_info.get('Values', {})
    
    # Check if it has a range (numerical)
    has_range = 'range' in values_info
    # Check if it has specific categorical values (non-range keys)
    has_categorical_values = any(key != 'range' for key in values_info.keys())
    
    if has_range and has_categorical_values:
        # Mixed variable: has both range and categorical values
        var_type = 'mixed'
        range_info = values_info.get('range', [])
        categorical_values = {k: v for k, v in values_info.items() if k != 'range'}
        
        return {
            'type': var_type,
            'description': var_info.get('Description', 'No description'),
            'range': range_info,
            'categorical_values': categorical_values,
            'total_categories': len(categorical_values),
            'is_numerical': True,  # Can be treated as numerical for analysis
            'special_values': list(categorical_values.keys())  # Special codes like 88, 99, 9999
        }
        
    elif has_range and not has_categorical_values:
        # Pure numerical variable
        var_type = 'numerical'
        range_info = values_info.get('range', [])
        
        return {
            'type': var_type,
            'description': var_info.get('Description', 'No description'),
            'range': range_info,
            'min_value': range_info[0] if len(range_info) >= 2 else None,
            'max_value': range_info[1] if len(range_info) >= 2 else None,
            'is_numerical': True,
            'special_values': []
        }
        
    elif has_categorical_values and not has_range:
        # Pure categorical variable
        var_type = 'categorical'
        categorical_values = values_info
        
        return {
            'type': var_type,
            'description': var_info.get('Description', 'No description'),
            'categorical_values': categorical_values,
            'total_categories': len(categorical_values),
            'is_numerical': False,
            'special_values': []
        }
        
    else:
        # No values information - treat as unknown
        return {
            'type': 'unknown',
            'description': var_info.get('Description', 'No description'),
            'is_numerical': False,
            'special_values': []
        }

def _print_categorization_summary(categorized_vars: Dict[str, Dict[str, Any]]) -> None:
    """Print summary of variable categorization results."""
    type_counts = {}
    for var_info in categorized_vars.values():
        var_type = var_info['type']
        type_counts[var_type] = type_counts.get(var_type, 0) + 1
    
    ui.print_info(f"📊 Variable categorization summary:")
    ui.print_info(f"   • Categorical: {type_counts.get('categorical', 0)}")
    ui.print_info(f"   • Numerical: {type_counts.get('numerical', 0)}")
    ui.print_info(f"   • Mixed: {type_counts.get('mixed', 0)}")
    ui.print_info(f"   • Unknown: {type_counts.get('unknown', 0)}")
    ui.print_info(f"   • Total: {len(categorized_vars)}")

def categorize_variables_from_dictionary(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Categorize variables from JSON dictionary as categorical, numerical, or mixed.
    
    Args:
        json_path (str): Path to the JSON dictionary file
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with variable categorization info
    """
    with open(json_path, 'r') as f:
        dictionary = json.load(f)
    
    categorized_vars = {}
    
    for var_name, var_info in dictionary.items():
        categorized_vars[var_name] = _categorize_single_variable(var_name, var_info)
    
    _print_categorization_summary(categorized_vars)
    return categorized_vars

# =============================================================================
# UNKNOWN VALUE MAPPING FUNCTIONS
# =============================================================================


def extract_unknown_codes_from_json(json_path: str) -> Dict[str, List[str]]:
    """
    Extract unknown/missing codes from JSON dictionary.
    
    Args:
        json_path (str): Path to JSON dictionary file
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping variable names to their unknown codes
    """
    unknown_codes = {}
    
    try:
        with open(json_path, 'r') as f:
            dictionary = json.load(f)
        
        # Keywords that indicate unknown/missing values
        unknown_keywords = ['unknown', 'not applicable', 'not available', 'refused', 'missing', 'n/a']
        
        for var_name, var_info in dictionary.items():
            values_info = var_info.get('Values', {})
            codes = []
            
            for value, description in values_info.items():
                if value == 'range':  # Skip range entries
                    continue
                
                # Check if description contains unknown keywords
                if isinstance(description, str):
                    description_lower = description.lower()
                    if any(keyword in description_lower for keyword in unknown_keywords):
                        codes.append(str(value))
            
            if codes:
                unknown_codes[var_name] = codes
        
        ui.print_info(f"🔍 Extracted unknown codes for {len(unknown_codes)} variables from JSON")
        
    except Exception as e:
        ui.print_warning(f"⚠️ Could not read JSON file: {e}")
    
    return unknown_codes

def map_unknown_values_to_nan(df: pd.DataFrame, relevant_variables: Dict[str, Dict[str, Any]], json_path: str = None) -> pd.DataFrame:
    """
    Replace unknown/missing codes with NaN for categorical variables only, using codes extracted from JSON.
    
    Args:
        df (pd.DataFrame): Input dataframe
        relevant_variables (Dict[str, Dict[str, Any]]): Variable categorization information
        json_path (str): Path to JSON dictionary file
    
    Returns:
        pd.DataFrame: Dataframe with unknown values replaced with NaN (categorical columns only)
    """
    processed_df = df.copy()
    
    ui.print_info(f"🔧 Replacing unknown/missing codes with NaN for categorical variables...")
    
    # Extract unknown codes from JSON if path provided
    unknown_codes_dict = {}
    if json_path:
        unknown_codes_dict = extract_unknown_codes_from_json(json_path)
    
    for col in processed_df.columns:
        # Find matching variable in relevant_variables to check if it's categorical
        var_info = None
        var_name = None
        for name, info in relevant_variables.items():
            if col.upper() == name.upper():
                var_info = info
                var_name = name
                break
        
        # Only process categorical variables
        if var_info and var_info.get('type') == 'categorical':
            # Find matching variable in JSON unknown codes
            json_var_name = None
            for json_name in unknown_codes_dict.keys():
                if col.upper() == json_name.upper():
                    json_var_name = json_name
                    break
            
            if json_var_name and json_var_name in unknown_codes_dict:
                codes_to_replace = unknown_codes_dict[json_var_name]
                # Check if any of these codes exist in the column
                found_codes = []
                for code in codes_to_replace:
                    if code in processed_df[col].values:
                        found_codes.append(code)
                
                if found_codes:
                    # Count how many values will be replaced
                    total_replacements = 0
                    for code in found_codes:
                        count = (processed_df[col] == code).sum()
                        total_replacements += count
                    
                    ui.print_info(f"   📊 Categorical column {col}:")
                    ui.print_info(f"      🔍 Unknown codes found: {found_codes}")
                    ui.print_info(f"      🔄 Replacing {total_replacements} values with NaN")
                    processed_df[col] = processed_df[col].replace(found_codes, np.nan)
    
    return processed_df

# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

def analyze_missing_data(df: pd.DataFrame, missing_threshold: float = 0.10) -> Dict[str, Any]:
    """
    Analyze missing data patterns in the dataframe.
    This is for reporting purposes only - no data is removed.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_threshold (float): Threshold for identifying high missingness (for reporting only)
    
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

def print_missing_analysis(analysis: Dict[str, Any]) -> None:
    """Print missing data analysis results."""
    # Commented out verbose missing data analysis output
    # ui.print_info(f"   📊 Total missing values: {analysis['total_missing']:,} ({analysis['overall_missing_pct']:.1f}% of all data)")
    # ui.print_info(f"   📊 Columns with high missingness (>10%): {len(analysis['high_missing_cols'])}")
    
    # if analysis['high_missing_cols']:
    #     ui.print_info(f"   📊 Top 10 columns with highest missingness:")
    #     top_missing = analysis['missing_percentages'].nlargest(10)
    #     for col, pct in top_missing.items():
    #         ui.print_info(f"     • {col}: {pct:.1f}% missing")
    pass

def identify_column_types_with_categorization(df: pd.DataFrame, relevant_variables: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify column types using JSON-based categorization information.
    
    Args:
        df (pd.DataFrame): Input dataframe
        relevant_variables (Dict[str, Dict[str, Any]]): Variable categorization information
    
    Returns:
        Tuple[List[str], List[str], List[str]]: (categorical_cols, numerical_cols, mixed_cols)
    """
    json_categorical_cols = []
    json_numerical_cols = []
    json_mixed_cols = []
    
    for col in df.columns:
        # Find the corresponding variable info
        var_info = None
        for var_name, info in relevant_variables.items():
            if col.upper() == var_name.upper() or col in info.get('description', ''):
                var_info = info
                break
        
        if var_info:
            var_type = var_info.get('type', 'unknown')
            is_numerical = var_info.get('is_numerical', False)
            
            if var_type == 'categorical':
                json_categorical_cols.append(col)
            elif var_type == 'numerical':
                json_numerical_cols.append(col)
            elif var_type == 'mixed':
                json_mixed_cols.append(col)
                # Mixed variables can be treated as numerical for analysis
                json_numerical_cols.append(col)
            else:
                # Unknown type - use pandas inference as fallback
                if df[col].dtype == 'object':
                    json_categorical_cols.append(col)
                else:
                    json_numerical_cols.append(col)
        else:
            # No categorization info - use pandas inference as fallback
            if df[col].dtype == 'object':
                json_categorical_cols.append(col)
            else:
                json_numerical_cols.append(col)
    
    return json_categorical_cols, json_numerical_cols, json_mixed_cols

def process_data_with_dictionary_mapping(df: pd.DataFrame, relevant_variables: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Process dataframe by handling unknown values and creating dummy variables, then renaming columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        relevant_variables (Dict[str, Dict[str, Any]]): Variable information with descriptions and value mappings
    
    Returns:
        pd.DataFrame: Processed dataframe with descriptive names and proper dummy variables
    """
    processed_df = df.copy()
    
    # Step 1: Replace unknown codes with NaN for each variable (using original column names)
    ui.print_info("🔧 Replacing unknown codes with NaN...")
    total_replacements = 0
    for var_name, var_info in relevant_variables.items():
        ui.print_info(f"   🔍 Processing variable: {var_name}")
        
        # Find the column by original name (case-insensitive)
        col_name = None
        for col in processed_df.columns:
            if col.upper() == var_name.upper():
                col_name = col
                break
        
        ui.print_info(f"      📍 Column name: {col_name}")
        
        if col_name and col_name in processed_df.columns:
            # Get unknown codes from the variable info
            unknown_codes = var_info.get('unknown_codes', [])
            ui.print_info(f"      🚫 Unknown codes to replace: {unknown_codes}")
            
            if unknown_codes:
                # Convert column to string for comparison
                col_series = processed_df[col_name].astype(str)
                # Replace unknown codes with NaN
                for code in unknown_codes:
                    mask = col_series == str(code)
                    if mask.any():
                        count = mask.sum()
                        processed_df.loc[mask, col_name] = np.nan
                        ui.print_info(f"      🔄 Replaced {count} instances of code '{code}' with NaN")
                        total_replacements += count
                    else:
                        ui.print_info(f"      ℹ️ Code '{code}' not found in data")
            else:
                ui.print_info(f"      ℹ️ No unknown codes defined for this variable")
        else:
            ui.print_info(f"      ⚠️ Column '{col_name}' not found in dataframe")
    
    ui.print_info(f"   📊 Total unknown codes replaced: {total_replacements}")
    
    # Step 2: Create dummy variables for categorical variables (using original column names)
    ui.print_info("🔧 Creating dummy variables for categorical variables...")
    total_dummies_created = 0
    dummy_variable_mappings = {}  # Store mappings for dummy variable renaming
    
    for var_name, var_info in relevant_variables.items():
        ui.print_info(f"   🔍 Processing variable: {var_name}")
        
        # Find the column by original name (case-insensitive)
        col_name = None
        for col in processed_df.columns:
            if col.upper() == var_name.upper():
                col_name = col
                break
        
        ui.print_info(f"      📍 Column name: {col_name}")
        
        if col_name and col_name in processed_df.columns:
            var_type = var_info.get('type', 'unknown')
            ui.print_info(f"      📊 Variable type: {var_type}")
            
            if var_type.lower() == 'categorical':
                ui.print_info(f"      ✅ Variable is categorical - proceeding with dummy creation")
                
                # Get value mappings from the original variable info
                value_mappings = var_info.get('Values', {})
                ui.print_info(f"      🗺️ Value mappings available: {len(value_mappings)} codes")
                if value_mappings:
                    ui.print_info(f"         Sample mappings: {dict(list(value_mappings.items())[:3])}")
                
                # Create a copy for dummy variable creation
                col_for_dummies = processed_df[col_name].copy()
                ui.print_info(f"      📈 Original unique values: {col_for_dummies.nunique()}")
                
                # Map numeric codes to descriptive names if available
                if value_mappings:
                    ui.print_info(f"      🔄 Mapping numeric codes to descriptive names...")
                    ui.print_info(f"         🗺️ Value mappings: {value_mappings}")
                    ui.print_info(f"         📊 Original values before mapping: {col_for_dummies.value_counts().to_dict()}")
                    col_for_dummies = col_for_dummies.astype(str)
                    # Handle float values that become "0.0", "1.0", etc. by converting to int strings
                    col_for_dummies = col_for_dummies.apply(lambda x: str(int(float(x))) if x.replace('.', '').replace('-', '').isdigit() else x)
                    col_for_dummies = col_for_dummies.map(value_mappings)
                    # Keep NaN values as NaN (don't fill with original values to avoid reintroducing unknown codes)
                    ui.print_info(f"         📊 Values after mapping: {col_for_dummies.value_counts().to_dict()}")
                    ui.print_info(f"      📈 Unique values after mapping: {col_for_dummies.nunique()}")
                
                # Check if we have reasonable number of unique values
                unique_count = col_for_dummies.nunique()
                if unique_count > 1 and unique_count <= 20:
                    # Show original values
                    original_values = col_for_dummies.value_counts().to_dict()
                    ui.print_info(f"      📊 Creating dummy variables for {col_name}:")
                    ui.print_info(f"         🔍 Value distribution: {original_values}")
                    
                    # Create dummy variables, excluding NaN values
                    dummies = pd.get_dummies(col_for_dummies, prefix=col_name, dummy_na=False)
                    dummies = dummies.astype(int)
                    
                    # Show created dummy variables
                    dummy_names = list(dummies.columns)
                    ui.print_info(f"         ✅ Created {len(dummy_names)} dummy variables: {dummy_names}")
                    total_dummies_created += len(dummy_names)
                    
                    # Store mapping for later renaming
                    dummy_variable_mappings[col_name] = {
                        'dummy_names': dummy_names,
                        'description': var_info.get('description', col_name)
                    }
                    
                    # Add dummy variables to dataframe
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    # Remove original categorical column
                    processed_df.drop(col_name, axis=1, inplace=True)
                    ui.print_info(f"         🗑️ Removed original column: {col_name}")
                else:
                    ui.print_info(f"      ⚠️ Skipping dummy creation (unique values: {unique_count})")
            else:
                ui.print_info(f"      ℹ️ Variable is not categorical - skipping dummy creation")
        else:
            ui.print_info(f"      ⚠️ Column '{col_name}' not found in dataframe")
    
    ui.print_info(f"   📊 Total dummy variables created: {total_dummies_created}")
    
    # Step 3: Rename columns to descriptions
    ui.print_info("🔧 Renaming columns to descriptions...")
    column_mapping = {}
    
    for var_name, var_info in relevant_variables.items():
        if 'description' in var_info:
            description = var_info['description']
            # Find matching column in dataframe (case-insensitive)
            matching_col = None
            for col in processed_df.columns:
                if col.upper() == var_name.upper():
                    matching_col = col
                    break
            
            if matching_col and matching_col != description:
                column_mapping[matching_col] = description
                ui.print_info(f"   📝 {matching_col} → {description}")
    
    # Apply column name mapping
    processed_df = processed_df.rename(columns=column_mapping)
    ui.print_info(f"   ✅ Column renaming completed!")
    
    # Step 4: Rename dummy variables to include actual category names
    ui.print_info("🔧 Renaming dummy variables to include actual category names...")
    dummy_rename_mapping = {}
    
    for original_col, info in dummy_variable_mappings.items():
        description = info['description']
        dummy_names = info['dummy_names']
        
        # Get the value mappings for this variable to convert codes to actual names
        var_info = None
        for var_name, var_data in relevant_variables.items():
            if var_name.upper() == original_col.upper():
                var_info = var_data
                break
        
        if var_info and 'Values' in var_info:
            value_mappings = var_info['Values']
            
            for dummy_name in dummy_names:
                # Extract the category part (everything after the prefix)
                category_code = dummy_name.replace(f"{original_col}_", "")
                
                # Convert the code to actual category name
                actual_category_name = value_mappings.get(category_code, category_code)
                
                # Create new name with description and actual category name
                new_name = f"{description}_{actual_category_name}"
                dummy_rename_mapping[dummy_name] = new_name
                ui.print_info(f"   📝 {dummy_name} → {new_name}")
        else:
            # Fallback if no value mappings available
            for dummy_name in dummy_names:
                category_part = dummy_name.replace(f"{original_col}_", "")
                new_name = f"{description}_{category_part}"
                dummy_rename_mapping[dummy_name] = new_name
                ui.print_info(f"   📝 {dummy_name} → {new_name}")
    
    # Apply dummy variable renaming
    processed_df = processed_df.rename(columns=dummy_rename_mapping)
    ui.print_info(f"   ✅ Dummy variable renaming completed!")
    
    return processed_df

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
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

def preprocess_data_with_categorization(df: pd.DataFrame, relevant_variables: Dict[str, Dict[str, Any]], missing_threshold: float = 0.10, json_path: str = None) -> Dict[str, Any]:
    """
    Perform data preprocessing using JSON-based categorization information.
    Preserves all missing data - no columns or rows are removed.
    
    Args:
        df (pd.DataFrame): Input dataframe to preprocess
        relevant_variables (Dict[str, Dict[str, Any]]): Variable categorization information
        missing_threshold (float): Threshold for identifying high missingness (for reporting only, no removal)
        json_path (str): Path to JSON dictionary for unknown value extraction (optional)
    
    Returns:
        Dict[str, Any]: Dictionary containing preprocessed dataframe and metadata (all missing data preserved)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for data preprocessing operations. Please install pandas.")
    
    ui.print_info(f"🔧 Starting data preprocessing with JSON-based categorization...")
    original_shape = df.shape
    ui.print_info(f"📊 Original data shape: {original_shape[0]} rows, {original_shape[1]} columns")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    ui.print_info(f"📊 Columns in processed dataframe: {processed_df.columns}")
    
    # 1. Analyze missing data patterns (silent analysis)
    missing_analysis = analyze_missing_data(processed_df)
    print_missing_analysis(missing_analysis)
    
    # 2. Preserve all missing data (no column removal)
    high_missing_cols = missing_analysis['high_missing_cols']
    
    # Commented out verbose missing data output
    # if high_missing_cols:
    #     ui.print_info(f"   📈 Found {len(high_missing_cols)} columns with high missingness (>{missing_threshold*100:.1f}%):")
    #     for col in high_missing_cols:
    #         missing_pct = missing_analysis['missing_percentages'][col]
    #         ui.print_info(f"   • {col}: {missing_pct:.1f}% missing (preserved)")
    #         ui.print_info(f"   ✅ All columns preserved - missing data will be handled by analysis methods")
    # else:
    #     ui.print_info(f"   ✅ No columns exceed the {missing_threshold*100:.1f}% missing threshold")
    
    # 3. Process data with dictionary mapping (replace names, handle unknown values, create dummies)
    ui.print_info(f"🔧 Processing data with dictionary mapping...")
    processed_df = process_data_with_dictionary_mapping(processed_df, relevant_variables)
    
    # 4. Convert remaining object columns to numeric if possible
    ui.print_info(f"🔧 Converting remaining object columns to numeric...")
    processed_df = convert_to_numeric(processed_df)
    
    # 5. Final data validation and warning about NaN values
    ui.print_info(f"🔍 Final data validation...")
    final_missing = processed_df.isnull().sum().sum()
    if final_missing > 0:
        ui.print_warning(f"⚠️ Dataset contains {final_missing} missing values (NaN)")
        ui.print_warning(f"⚠️ IMPORTANT: For each specific test, remove rows with NaN values in the columns being compared")
        ui.print_warning(f"⚠️ Example: If comparing columns A and B, drop rows where either A or B has NaN values")
    
    final_shape = processed_df.shape
    ui.print_success(f"✅ Data preprocessing with categorization completed!")
    ui.print_info(f"📊 Final data shape: {final_shape[0]} rows, {final_shape[1]} columns")
    ui.print_info(f"📈 Columns removed: 0 (all missing data preserved)")
    ui.print_info(f"📈 Columns added: {final_shape[1] - original_shape[1]} (dummy variables only)")
    
    # Summary of processing
    ui.print_info(f"📋 Summary of data processing:")
    ui.print_info(f"   📊 Total dummy variables created: {final_shape[1] - original_shape[1]}")
    ui.print_info(f"   ❓ Total missing values (NaN): {final_missing}")
    
    # Return dictionary with preprocessed data and metadata
    return {
        'dataframe': processed_df,
        'original_shape': original_shape,
        'final_shape': final_shape,
        'categorical_columns': [],  # No longer tracking separately
        'numerical_columns': [],    # No longer tracking separately
        'mixed_columns': [],        # No longer tracking separately
        'missing_values_count': final_missing,
        'columns_removed': 0,  # No columns removed - all missing data preserved
        'columns_added': final_shape[1] - original_shape[1]  # Only dummy variables added
    }

# =============================================================================
# VARIABLE SELECTION FUNCTIONS
# =============================================================================

def build_llm_prompt_for_variable_selection(hypothesis: str, dictionary: Dict[str, Any], categorized_vars: Dict[str, Dict[str, Any]], csv_statistics: Dict[str, Any] = None, previous_iteration_context: str = None, previously_used_variables: List[str] = None) -> str:
    """Build the LLM prompt for variable selection."""
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
    
    # Create a formatted string of all keys with descriptions and categorization
    keys_info = []
    for key, info in dictionary.items():
        description = info.get('Description', 'No description available')
        var_cat = categorized_vars.get(key, {})
        var_type = var_cat.get('type', 'unknown')
        is_numerical = var_cat.get('is_numerical', False)
        
        # Add categorization info to the description
        type_info = f" [{var_type.upper()}]" if var_type != 'unknown' else ""
        numerical_info = " [NUMERICAL]" if is_numerical else " [CATEGORICAL]"
        
        keys_info.append(f"Key: {key} | Description: {description}{type_info}{numerical_info}")
    
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
    
    return prompt

def parse_llm_response_for_variables(response: str) -> List[str]:
    """Parse LLM response to extract variable names."""
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
    
    return variable_names

def get_relevant_variables(model_client: ModelClient, hypothesis: str, json_path: str, csv_statistics: Dict[str, Any] = None, previous_iteration_context: str = None, previously_used_variables: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get all relevant variables for a hypothesis from a JSON dictionary.
    
    This function uses an LLM to analyze a hypothesis and identify all relevant
    variables from a data dictionary. It returns a dictionary with variable names
    as keys and their categorization information as values.
    
    Args:
        model_client (ModelClient): Model client for LLM interactions
        hypothesis (str): The hypothesis to analyze
        json_path (str): Path to the JSON dictionary file
        csv_statistics (Dict[str, Any], optional): Statistics from the CSV file
        previous_iteration_context (str, optional): Context from previous outer loop iteration
        previously_used_variables (List[str], optional): List of variables used in previous iterations
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with variable names as keys and categorization info as values
    """
    
    # First, categorize all variables from the dictionary
    ui.print_info(f"🔍 Categorizing variables from dictionary...")
    categorized_vars = categorize_variables_from_dictionary(json_path)
    
    # Load the JSON dictionary for the LLM prompt
    with open(json_path, 'r') as f:
        dictionary = json.load(f)
    
    # Build the LLM prompt
    prompt = build_llm_prompt_for_variable_selection(
        hypothesis, dictionary, categorized_vars, csv_statistics, 
        previous_iteration_context, previously_used_variables
    )

    # Generate response using the model client
    messages = [{"role": "user", "content": prompt}]
    response = model_client.generate(messages)
    print(f"Data Processing Response: {response}")
    
    # Parse the response to extract variable names
    variable_names = parse_llm_response_for_variables(response)
    
    # If Python list parsing failed, return empty dictionary
    if not variable_names:
        print("No variable names found in Python list, returning empty dictionary")
        return {}
    
    # Create result dictionary with variable names and full categorization info
    result_dict = {}
    for var_name in variable_names:
        if var_name in categorized_vars:
            result_dict[var_name] = categorized_vars[var_name]
        elif var_name in dictionary:
            # Fallback for variables not in categorized_vars
            result_dict[var_name] = {
                'type': 'unknown',
                'description': dictionary[var_name].get('Description', 'No description available'),
                'is_numerical': False,
                'special_values': []
            }
    
    ui.print_info(f"🎯 Selected {len(result_dict)} variables with categorization:")
    for var_name, var_info in result_dict.items():
        var_type = var_info.get('type', 'unknown')
        is_numerical = var_info.get('is_numerical', False)
        ui.print_info(f"   • {var_name}: {var_type} ({'numerical' if is_numerical else 'categorical'})")
    
    return result_dict

# =============================================================================
# FILE HANDLING FUNCTIONS
# =============================================================================

def detect_dataset_files(folder_path: str) -> Tuple[str, str]:
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

def get_dataset_statistics(csv_path: str) -> Dict[str, Any]:
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

# =============================================================================
# MAIN DATA PROCESSOR CLASS (SIMPLIFIED ORCHESTRATOR)
# =============================================================================

class DataProcessor:
    """Simplified orchestrator for data processing operations."""
    
    def __init__(self, model_client: ModelClient):
        """Initialize the data processor with a model client."""
        self.model_client = model_client
    
    def categorize_variables_from_dictionary(self, json_path: str) -> Dict[str, Dict[str, Any]]:
        """Wrapper for the modular categorization function."""
        return categorize_variables_from_dictionary(json_path)
    
    def get_relevant_variables(self, hypothesis: str, json_path: str, csv_statistics: Dict[str, Any] = None, previous_iteration_context: str = None, previously_used_variables: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Wrapper for the modular variable selection function."""
        return get_relevant_variables(self.model_client, hypothesis, json_path, csv_statistics, previous_iteration_context, previously_used_variables)
    
    def preprocess_data_with_categorization(self, df: pd.DataFrame, relevant_variables: Dict[str, Dict[str, Any]], missing_threshold: float = 0.10, json_path: str = None) -> Dict[str, Any]:
        """Wrapper for the modular preprocessing function."""
        return preprocess_data_with_categorization(df, relevant_variables, missing_threshold, json_path)
    
    def get_dataset_statistics(self, csv_path: str) -> Dict[str, Any]:
        """Wrapper for the modular statistics function."""
        return get_dataset_statistics(csv_path)
    
    def detect_dataset_files(self, folder_path: str) -> Tuple[str, str]:
        """Wrapper for the modular file detection function."""
        return detect_dataset_files(folder_path)
    
    def filter_csv_with_dictionary(self, folder_path: str, relevant_variables: Dict[str, Dict[str, Any]], hypothesis: str) -> Tuple[str, Dict[str, str], pd.DataFrame]:
        """
        Filter CSV files in a folder using relevant variables from dictionary, preprocess the data, and save to .working.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            relevant_variables (Dict[str, Dict[str, Any]]): Dictionary of relevant variable names and categorization info
            hypothesis (str): The hypothesis being tested
        
        Returns:
            tuple[str, Dict[str, str], pd.DataFrame]: Path to the filtered CSV file, dictionary of variable datatypes, and preprocessed dataframe
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
                    for var_name, var_info in relevant_variables.items():
                        if col.upper() == var_name.upper():
                            # Use the description as the new column name
                            description = var_info.get('description', 'No description')
                            # Remove special characters and replace spaces with underscores for variable names
                            clean_description = re.sub(r"[^\w\s]", "", description)  # Remove special characters
                            clean_description = re.sub(r"\s+", "_", clean_description.strip())  # Replace spaces with underscores
                            column_mapping[col] = clean_description
                            break
                
            # Preprocess the data using JSON-based categorization (create dummy variables, etc.)
            # Get JSON path for unknown value mapping
            json_path = os.path.join(folder_path, [f for f in os.listdir(folder_path) if f.lower().endswith('.json')][0])
            preprocessing_result = preprocess_data_with_categorization(filtered_df, relevant_variables, json_path=json_path)
            processed_df = preprocessing_result['dataframe']
            
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
            
            return str(output_path), variable_datatypes, processed_df
            
        except Exception as e:
            error_msg = f"Error filtering and preprocessing CSV file: {str(e)}"
            ui.print_error(error_msg, "csv_filtering")
            raise RuntimeError(error_msg) from e
    
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
            csv_path, json_path = detect_dataset_files(folder_path)
            
            if not hypothesis.strip():
                ui.print_warning(f"⚠️ No hypothesis provided. Exiting manual preprocessing.")
                return {'success': False, 'error': 'No hypothesis provided'}
            
            ui.print_info(f"🔬 Hypothesis: {hypothesis}")
            ui.print_info(f"📁 Dataset folder: {folder_path}")
            ui.print_info(f"📊 CSV file: {csv_path}")
            ui.print_info(f"📚 JSON dictionary: {json_path}")
            
            # Get CSV statistics
            csv_statistics = get_dataset_statistics(csv_path)
            
            # Get relevant variables from dictionary
            ui.print_info(f"🔍 Analyzing hypothesis to find relevant variables...")
            if previous_iteration_context:
                ui.print_info(f"🔄 Using previous iteration context to guide variable selection...")
            if previously_used_variables:
                ui.print_info(f"📋 Avoiding {len(previously_used_variables)} previously used variables: {', '.join(previously_used_variables[:5])}{'...' if len(previously_used_variables) > 5 else ''}")
            
            relevant_variables = get_relevant_variables(self.model_client, hypothesis, json_path, csv_statistics, previous_iteration_context, previously_used_variables)
            
            if not relevant_variables:
                ui.print_warning(f"⚠️ No relevant variables found. Using all available data.")
                # Create a dummy dictionary with all columns (we'll handle this in filtering)
                relevant_variables = {"ALL_COLUMNS": "All available columns"}
            
            ui.print_info(f"🎯 Found {len(relevant_variables)} relevant variables:")
            for var_name, var_info in relevant_variables.items():
                description = var_info.get('description', 'No description')
                var_type = var_info.get('type', 'unknown')
                is_numerical = var_info.get('is_numerical', False)
                ui.print_info(f"   • {var_name}: {description} [{var_type}] ({'numerical' if is_numerical else 'categorical'})")

            # Filter CSV files using relevant variables
            filtered_csv_path, variable_datatypes, preprocessed_df = self.filter_csv_with_dictionary(folder_path, relevant_variables, hypothesis)
            
            # Get statistics for the filtered dataset
            filtered_csv_statistics = get_dataset_statistics(filtered_csv_path)
            
            # Create enhanced variable information string with categorization
            variable_info_str = []
            for col, dtype in variable_datatypes.items():
                # Find the original variable info
                var_info = None
                for var_name, info in relevant_variables.items():
                    if col.upper() == var_name.upper() or col in info.get('description', ''):
                        var_info = info
                        break
                
                if var_info:
                    var_type = var_info.get('type', 'unknown')
                    is_numerical = var_info.get('is_numerical', False)
                    special_values = var_info.get('special_values', [])
                    special_info = f" (special values: {special_values})" if special_values else ""
                    variable_info_str.append(f"- {col}: {dtype} [{var_type}] ({'numerical' if is_numerical else 'categorical'}){special_info}")
                else:
                    variable_info_str.append(f"- {col}: {dtype}")
            
            variable_datatypes_str = "\n".join(variable_info_str)

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
                        f"- Categorical columns: {filtered_csv_statistics.get('categorical_cols', 'N/A')}\n\n"
                        f"IMPORTANT: For each specific test, remove rows with NaN values in the columns being compared. "
                        f"Example: If comparing columns A and B, drop rows where either A or B has NaN values."
                    )
                }]
                conversation_manager.messages.extend(summarized_messages)
            print(summarized_messages) # TODO: remove this
            ui.print_info(f"✅ Manual preprocessing completed successfully!")
            ui.print_info(f"📊 Filtered data saved to: {filtered_csv_path}")
            
            # Return the information needed for enhanced planning including preprocessed data dictionary
            return {
                'success': True,
                'hypothesis': hypothesis,
                'filtered_csv_statistics': filtered_csv_statistics,
                'relevant_variables': relevant_variables,
                'filtered_csv_path': filtered_csv_path,
                'preprocessed_data': {
                    'dataframe': preprocessed_df,
                    'variable_datatypes': variable_datatypes,
                    'categorical_columns': filtered_csv_statistics.get('categorical_column_names', []),
                    'numerical_columns': filtered_csv_statistics.get('numeric_column_names', []),
                    'missing_values_count': filtered_csv_statistics.get('missing_data', 0)
                }
            }
            
        except Exception as e:
            error_msg = f"Manual preprocessing failed: {str(e)}"
            ui.print_error(error_msg, "manual_preprocessing")
            return {
                'success': False,
                'error': error_msg
            }
