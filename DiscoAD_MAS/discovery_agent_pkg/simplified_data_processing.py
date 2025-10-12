"""
Simplified Data Processing for the Discovery Agent.

This module provides a streamlined approach to data processing with focused functions:
1. LLM-powered variable selection
2. Unknown value replacement with NaN
3. Categorical variable dummy encoding
4. Column renaming with descriptions
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


def query_llm_for_variable_selection(model_client: ModelClient, hypothesis: str, json_path: str) -> List[str]:
    """
    Query an LLM to filter columns and return relevant variable names.
    
    Args:
        model_client (ModelClient): Model client for LLM interactions
        hypothesis (str): The hypothesis to analyze
        json_path (str): Path to the JSON dictionary file
    
    Returns:
        List[str]: List of selected variable names
    """
    ui.print_info(f"🔍 Querying LLM for variable selection...")
    
    # Load the JSON dictionary
    with open(json_path, 'r') as f:
        dictionary = json.load(f)
    
    # Create a formatted string of all variables with descriptions
    variables_info = []
    for var_name, var_info in dictionary.items():
        description = var_info.get('Description', 'No description available')
        values = var_info.get('Values', {})
        
        # Determine if it's categorical or numerical
        has_range = 'range' in values
        has_categorical = any(key != 'range' for key in values.keys())
        
        if has_range and has_categorical:
            var_type = "Mixed (numerical with special codes)"
        elif has_range:
            var_type = "Numerical"
        elif has_categorical:
            var_type = "Categorical"
        else:
            var_type = "Unknown"
        
        variables_info.append(f"• {var_name}: {description} [{var_type}]")
    
    variables_text = "\n".join(variables_info)
    
    # Create the prompt for the LLM
    prompt = f"""You are a data analysis expert. Given a hypothesis and a dictionary of data fields, identify the most useful variables for testing this hypothesis.

Hypothesis: "{hypothesis}"

Available variables:
{variables_text}

Please select the most relevant variables for this hypothesis. Focus on:
1. Variables directly related to the hypothesis
2. Essential demographic/control variables (age, sex, education)
3. Key medical/clinical variables
4. Avoid redundant or highly correlated variables

Provide your response as a Python list of variable names:
variable_names = ["VAR1", "VAR2", "VAR3", ...]

Aim for 10-20 variables maximum for a focused analysis."""

    # Generate response using the model client
    messages = [{"role": "user", "content": prompt}]
    response = model_client.generate(messages)
    ui.print_info(f"LLM Response: {response}")
    
    # Parse the response to extract variable names
    variable_names = []
    
    # Try to extract from Python list format
    python_list_pattern = r'variable_names\s*=\s*\[(.*?)\]'
    python_list_match = re.search(python_list_pattern, response, re.DOTALL)
    
    if python_list_match:
        try:
            list_content = python_list_match.group(1)
            # Extract quoted strings
            quoted_strings = re.findall(r'"([^"]*)"', list_content)
            if quoted_strings:
                variable_names = quoted_strings
            else:
                # Try single quotes
                single_quoted_strings = re.findall(r"'([^']*)'", list_content)
                if single_quoted_strings:
                    variable_names = single_quoted_strings
        except Exception as e:
            ui.print_warning(f"Failed to parse Python list: {e}")
    
    if not variable_names:
        ui.print_warning("No variable names found in LLM response. Using all available variables.")
        variable_names = list(dictionary.keys())
    
    ui.print_info(f"🎯 Selected {len(variable_names)} variables: {variable_names}")
    return variable_names


def replace_unknown_values_with_nan(df: pd.DataFrame, json_path: str) -> pd.DataFrame:
    """
    Replace unknown/missing values with NaN across all columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        json_path (str): Path to JSON dictionary file
    
    Returns:
        pd.DataFrame: Dataframe with unknown values replaced with NaN
    """
    ui.print_info(f"🔧 Replacing unknown values with NaN...")
    
    # Load the JSON dictionary
    with open(json_path, 'r') as f:
        dictionary = json.load(f)
    
    processed_df = df.copy()
    total_replacements = 0
    
    # Keywords that indicate unknown/missing values
    unknown_keywords = ['unknown', 'not applicable', 'not available', 'refused', 'missing', 'n/a']
    
    for col in processed_df.columns:
        # Find matching variable in dictionary (case-insensitive)
        var_info = None
        for var_name, info in dictionary.items():
            if col.upper() == var_name.upper():
                var_info = info
                break
        
        if var_info and 'Values' in var_info:
            values_info = var_info['Values']
            unknown_codes = []
            
            # Find unknown codes
            for value, description in values_info.items():
                if value == 'range':  # Skip range entries
                    continue
                
                if isinstance(description, str):
                    description_lower = description.lower()
                    if any(keyword in description_lower for keyword in unknown_keywords):
                        unknown_codes.append(str(value))
            
            # Replace unknown codes with NaN
            if unknown_codes:
                col_replacements = 0
                for code in unknown_codes:
                    if code in processed_df[col].values:
                        mask = processed_df[col] == code
                        count = mask.sum()
                        processed_df.loc[mask, col] = np.nan
                        col_replacements += count
                        ui.print_info(f"   📊 {col}: Replaced {count} instances of '{code}' with NaN")
                
                total_replacements += col_replacements
                if col_replacements > 0:
                    ui.print_info(f"   ✅ {col}: Total replacements = {col_replacements}")
    
    ui.print_info(f"📊 Total unknown values replaced: {total_replacements}")
    return processed_df


def create_dummy_variables_for_categorical(df: pd.DataFrame, json_path: str) -> pd.DataFrame:
    """
    Create dummy variables for categorical columns (one per category, excluding unknown).
    
    Args:
        df (pd.DataFrame): Input dataframe
        json_path (str): Path to JSON dictionary file
    
    Returns:
        pd.DataFrame: Dataframe with dummy variables created
    """
    ui.print_info(f"🔧 Creating dummy variables for categorical columns...")
    
    # Load the JSON dictionary
    with open(json_path, 'r') as f:
        dictionary = json.load(f)
    
    processed_df = df.copy()
    dummy_variable_mappings = {}  # Store mappings for later renaming
    
    for col in processed_df.columns:
        # Find matching variable in dictionary (case-insensitive)
        var_info = None
        for var_name, info in dictionary.items():
            if col.upper() == var_name.upper():
                var_info = info
                break
        
        if var_info and 'Values' in var_info:
            values_info = var_info['Values']
            
            # Check if it's truly categorical (has specific values but NO range)
            has_range = 'range' in values_info
            has_categorical_values = any(key != 'range' for key in values_info.keys())
            is_pure_categorical = has_categorical_values and not has_range
            
            if is_pure_categorical:
                ui.print_info(f"   🔍 Processing categorical column: {col}")
                
                # Get categorical values (exclude range and unknown values)
                categorical_values = {}
                unknown_keywords = ['unknown', 'not applicable', 'not available', 'refused', 'missing', 'n/a']
                
                for value, description in values_info.items():
                    if value == 'range':  # Skip range entries
                        continue
                    
                    # Skip unknown values
                    if isinstance(description, str):
                        description_lower = description.lower()
                        if any(keyword in description_lower for keyword in unknown_keywords):
                            continue
                    
                    categorical_values[value] = description
                
                if len(categorical_values) > 1:  # Need at least 2 categories for dummy variables
                    ui.print_info(f"      📊 Creating dummy variables for {len(categorical_values)} categories")
                    
                    # Map numeric codes to descriptive names
                    col_for_dummies = processed_df[col].astype(str)
                    # Handle float values that become "0.0", "1.0", etc. by converting to int strings
                    col_for_dummies = col_for_dummies.apply(lambda x: str(int(float(x))) if x.replace('.', '').replace('-', '').isdigit() else x)
                    col_for_dummies = col_for_dummies.map(categorical_values)
                    # Keep NaN values as NaN (don't fill with original values to avoid reintroducing unknown codes)
                    
                    # Create dummy variables, excluding NaN values
                    dummies = pd.get_dummies(col_for_dummies, prefix=col, dummy_na=False)
                    dummies = dummies.astype(int)
                    
                    # Store mapping for later renaming
                    dummy_variable_mappings[col] = {
                        'dummy_names': list(dummies.columns),
                        'description': var_info.get('Description', col),
                        'value_mappings': categorical_values
                    }
                    
                    # Add dummy variables to dataframe
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    # Remove original categorical column
                    processed_df.drop(col, axis=1, inplace=True)
                    
                    ui.print_info(f"      ✅ Created {len(dummies.columns)} dummy variables: {list(dummies.columns)}")
                else:
                    ui.print_info(f"      ⚠️ Skipping {col} (insufficient categories: {len(categorical_values)})")
            else:
                # Skip numerical and mixed variables (they don't need dummy variables)
                pass
    
    # Store mappings for later use
    processed_df.attrs['dummy_mappings'] = dummy_variable_mappings
    
    return processed_df


def rename_columns_with_descriptions(df: pd.DataFrame, json_path: str) -> pd.DataFrame:
    """
    Rename columns to use variable descriptions instead of variable names.
    
    Args:
        df (pd.DataFrame): Input dataframe
        json_path (str): Path to JSON dictionary file
    
    Returns:
        pd.DataFrame: Dataframe with renamed columns
    """
    ui.print_info(f"🔧 Renaming columns with descriptions...")
    
    # Load the JSON dictionary
    with open(json_path, 'r') as f:
        dictionary = json.load(f)
    
    processed_df = df.copy()
    column_mapping = {}
    
    # Get dummy variable mappings if they exist
    dummy_mappings = processed_df.attrs.get('dummy_mappings', {})
    
    # First, rename dummy variables with descriptions and category names
    for original_col, info in dummy_mappings.items():
        description = info['description']
        dummy_names = info['dummy_names']
        value_mappings = info['value_mappings']
        
        for dummy_name in dummy_names:
            # Extract the category part (everything after the prefix)
            category_code = dummy_name.replace(f"{original_col}_", "")
            
            # Convert the code to actual category name
            actual_category_name = value_mappings.get(category_code, category_code)
            
            # Create new name with description and actual category name
            new_name = f"{description}_{actual_category_name}"
            column_mapping[dummy_name] = new_name
            ui.print_info(f"   📝 {dummy_name} → {new_name}")
    
    # Then, rename regular columns with descriptions
    for col in processed_df.columns:
        if col not in column_mapping:  # Skip already mapped dummy variables
            # Find matching variable in dictionary (case-insensitive)
            var_info = None
            for var_name, info in dictionary.items():
                if col.upper() == var_name.upper():
                    var_info = info
                    break
            
            if var_info and 'Description' in var_info:
                description = var_info['Description']
                # Clean description for use as column name
                clean_description = re.sub(r"[^\w\s]", "", description)  # Remove special characters
                clean_description = re.sub(r"\s+", "_", clean_description.strip())  # Replace spaces with underscores
                
                if col != clean_description:
                    column_mapping[col] = clean_description
                    ui.print_info(f"   📝 {col} → {clean_description}")
    
    # Apply column name mapping
    processed_df = processed_df.rename(columns=column_mapping)
    ui.print_info(f"   ✅ Column renaming completed!")
    
    return processed_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path where to save the CSV file
    """
    ui.print_info(f"💾 Saving processed data to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataframe
    df.to_csv(output_path, index=False)
    
    ui.print_success(f"✅ Processed data saved successfully!")
    ui.print_info(f"📊 Final data shape: {df.shape[0]} rows, {df.shape[1]} columns")


def simplified_data_preprocessing(model_client: ModelClient, hypothesis: str, csv_path: str, json_path: str, output_path: str = "working/processed_dataset.csv") -> pd.DataFrame:
    """
    Main function to perform simplified data preprocessing.
    
    Args:
        model_client (ModelClient): Model client for LLM interactions
        hypothesis (str): The hypothesis to analyze
        csv_path (str): Path to the CSV file
        json_path (str): Path to the JSON dictionary file
        output_path (str): Path where to save the processed data
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for data preprocessing operations. Please install pandas.")
    
    ui.print_info(f"🚀 Starting simplified data preprocessing...")
    ui.print_info(f"🔬 Hypothesis: {hypothesis}")
    ui.print_info(f"📊 CSV file: {csv_path}")
    ui.print_info(f"📚 JSON dictionary: {json_path}")
    
    # Step 1: Load the CSV data
    ui.print_info(f"📖 Loading CSV data...")
    df = pd.read_csv(csv_path)
    ui.print_info(f"📊 Original data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Query LLM to filter columns
    ui.print_info(f"🤖 Step 1: Querying LLM for variable selection...")
    selected_variables = query_llm_for_variable_selection(model_client, hypothesis, json_path)
    
    # Filter dataframe to only include selected variables
    available_columns = []
    for var_name in selected_variables:
        # Try exact match first
        if var_name in df.columns:
            available_columns.append(var_name)
        else:
            # Try case-insensitive match
            for col in df.columns:
                if col.upper() == var_name.upper():
                    available_columns.append(col)
                    break
    
    if available_columns:
        df = df[available_columns]
        ui.print_info(f"🎯 Filtered to {len(available_columns)} selected columns")
    else:
        ui.print_warning("⚠️ No selected columns found in data. Using all columns.")
    
    # Step 3: Replace unknown values with NaN
    ui.print_info(f"🔧 Step 2: Replacing unknown values with NaN...")
    df = replace_unknown_values_with_nan(df, json_path)
    
    # Step 4: Create dummy variables for categorical columns
    ui.print_info(f"🔧 Step 3: Creating dummy variables for categorical columns...")
    df = create_dummy_variables_for_categorical(df, json_path)
    
    # Step 5: Rename columns with descriptions
    ui.print_info(f"🔧 Step 4: Renaming columns with descriptions...")
    df = rename_columns_with_descriptions(df, json_path)
    
    # Step 6: Save processed data
    ui.print_info(f"💾 Step 5: Saving processed data...")
    save_processed_data(df, output_path)
    
    ui.print_success(f"✅ Simplified data preprocessing completed!")
    ui.print_info(f"📊 Final data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


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
# COMPATIBILITY WRAPPER CLASS
# =============================================================================

class SimplifiedDataProcessor:
    """Simplified data processor that maintains compatibility with the current pipeline."""
    
    def __init__(self, model_client: ModelClient):
        """Initialize with model client."""
        self.model_client = model_client
    
    def run_simplified_preprocessing(self, hypothesis: str, dataset_folder: str = None, 
                                   previous_iteration_context: str = None, 
                                   conversation_manager=None, 
                                   previously_used_variables: List[str] = None):
        """
        Run simplified preprocessing that maintains the same interface as the original.
        
        Args:
            hypothesis (str): The hypothesis to test
            dataset_folder (str): Path to folder containing CSV and JSON files
            previous_iteration_context (str, optional): Context from previous iteration
            conversation_manager: Conversation manager to add system messages
            previously_used_variables (List[str], optional): Previously used variables
        
        Returns:
            Dict[str, Any]: Result dictionary with same structure as original
        """
        ui.print_info(f"🔧 Starting simplified preprocessing...")
        
        try:
            # Auto-detect CSV and JSON files
            csv_path, json_path = detect_dataset_files(dataset_folder)
            
            # Run the simplified preprocessing
            output_path = "working/processed_dataset.csv"
            processed_df = simplified_data_preprocessing(
                model_client=self.model_client,
                hypothesis=hypothesis,
                csv_path=csv_path,
                json_path=json_path,
                output_path=output_path
            )
            
            # Get statistics for the processed dataset
            filtered_csv_statistics = get_dataset_statistics(output_path)
            
            # Create variable datatypes dictionary
            variable_datatypes = {}
            for col in processed_df.columns:
                dtype = str(processed_df[col].dtype)
                variable_datatypes[col] = dtype
            
            # Create enhanced variable information string
            variable_info_str = []
            for col, dtype in variable_datatypes.items():
                variable_info_str.append(f"- {col}: {dtype}")
            
            variable_datatypes_str = "\n".join(variable_info_str)
            
            # Add system message to conversation if conversation_manager is provided
            if conversation_manager:
                summarized_messages = [{
                    "role": "System",
                    "content": (
                        f"Processed data saved to: {output_path}\n\n"
                        f"Processed variables:\n{variable_datatypes_str}\n\n"
                        f"Processed dataset statistics:\n"
                        f"- Total rows: {filtered_csv_statistics.get('total_rows', 'N/A'):,}\n"
                        f"- Total columns: {filtered_csv_statistics.get('total_cols', 'N/A')}\n"
                        f"- Missing values: {filtered_csv_statistics.get('missing_data', 'N/A'):,} ({filtered_csv_statistics.get('missing_percentage', 'N/A')}%)\n"
                        f"- Numeric columns: {filtered_csv_statistics.get('numeric_cols', 'N/A')}\n"
                        f"- Categorical columns: {filtered_csv_statistics.get('categorical_cols', 'N/A')}\n\n"
                        f"IMPORTANT: For each specific test, remove rows with NaN values in the columns being compared."
                    )
                }]
                conversation_manager.messages.extend(summarized_messages)
            
            ui.print_success(f"✅ Simplified preprocessing completed successfully!")
            
            # Return the same structure as the original method
            return {
                'success': True,
                'hypothesis': hypothesis,
                'filtered_csv_statistics': filtered_csv_statistics,
                'relevant_variables': {},  # Simplified version doesn't track this separately
                'filtered_csv_path': output_path,
                'preprocessed_data': {
                    'dataframe': processed_df,
                    'variable_datatypes': variable_datatypes,
                    'categorical_columns': filtered_csv_statistics.get('categorical_column_names', []),
                    'numerical_columns': filtered_csv_statistics.get('numeric_column_names', []),
                    'missing_values_count': filtered_csv_statistics.get('missing_data', 0)
                }
            }
            
        except Exception as e:
            error_msg = f"Simplified preprocessing failed: {str(e)}"
            ui.print_error(error_msg, "simplified_preprocessing")
            return {
                'success': False,
                'error': error_msg
            }
    
    # Maintain compatibility with other methods that might be called
    def get_dataset_statistics(self, csv_path: str) -> Dict[str, Any]:
        """Wrapper for dataset statistics."""
        return get_dataset_statistics(csv_path)
    
    def detect_dataset_files(self, folder_path: str) -> Tuple[str, str]:
        """Wrapper for file detection."""
        return detect_dataset_files(folder_path)


# Example usage function
def run_simplified_preprocessing_example():
    """
    Example function showing how to use the simplified data preprocessing.
    """
    # This would be called from your main application
    # model_client = ModelClient(...)  # Initialize your model client
    # hypothesis = "Your hypothesis here"
    # csv_path = "path/to/your/data.csv"
    # json_path = "path/to/your/dictionary.json"
    # output_path = "working/processed_dataset.csv"
    
    # processed_df = simplified_data_preprocessing(
    #     model_client=model_client,
    #     hypothesis=hypothesis,
    #     csv_path=csv_path,
    #     json_path=json_path,
    #     output_path=output_path
    # )
    
    pass
