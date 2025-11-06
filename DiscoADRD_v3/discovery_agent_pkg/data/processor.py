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
from typing import Dict, Any, List, Tuple, Optional
from ..ui.interface import ui
from ..agents.models import ModelClient
from .cache import DataCache

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

def query_llm_for_variable_selection(model_client: ModelClient, hypothesis: str, data_cache: DataCache) -> List[str]:
    """
    Query an LLM to filter columns and return relevant variable names.
    
    Args:
        model_client (ModelClient): Model client for LLM interactions
        hypothesis (str): The hypothesis to analyze
        data_cache (DataCache): Data cache containing the JSON dictionary
    
    Returns:
        List[str]: List of selected variable names, or empty list if no relevant variables found
    """
    ui.print_info(f":mag: Querying LLM for variable selection...")
    
    # Get the JSON dictionary from cache
    dictionary = data_cache.get_json_dictionary()
    
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
            # Get categorical value descriptions (excluding 'range')
            cat_values = [values[k] for k in values.keys() if k != 'range']
            var_type_str = f"{var_type}: {', '.join(cat_values)}"
        elif has_range:
            var_type_str = "Numerical"
        elif has_categorical:
            var_type_str = "Categorical"
            # Get categorical value descriptions
            cat_values = list(values.values())
            var_type_str = f"{var_type_str}: {', '.join(cat_values)}"
        else:
            var_type_str = "Unknown"

        variables_info.append(f"• {var_name}: {description} [{var_type_str}]")
    
    variables_text = "\n".join(variables_info)
    
    # Create the prompt for the LLM
    prompt = f"""You are a data analysis expert. Given a hypothesis and a dictionary of data fields, identify the most useful variables for testing this hypothesis.

Hypothesis: "{hypothesis}"

Available variables:
{variables_text}

Please select the most relevant variables for this hypothesis. Focus on:
1. Variables directly related to the hypothesis
2. Essential demographic/control variables (age, sex, race, education, etc.)
3. Key medical/clinical variables mentioned or implied in the hypothesis
4. Variables needed to define comparison groups or subgroups
5. Ensure coverage of all key terms in the hypothesis. If a key term is a subgroup/category, include the variable that holds that category. If a term could belong to multiple parents, include both parents unless metadata clearly indicates one.
6. Avoid redundant or highly correlated variables only when they provide the same information

Provide your response as a Python list of variable names:
variable_names = ["VAR1", "VAR2", "VAR3", ...]

Aim for 10-20 variables maximum for a focused analysis."""

    # Generate response using the model client
    messages = [{"role": "user", "content": prompt}]
    response = model_client.generate(messages)
    ui.print_info(f"LLM Response: {response}")
    
    # Parse the response to extract variable names
    variable_names = []
    
    # Try to extract from Python list format - parsing from the end
    python_list_pattern = r'variable_names\s*=\s*\[(.*?)\]'
    # Find all matches
    all_matches = list(re.finditer(python_list_pattern, response, re.DOTALL))

    # Take the last match if any exist
    if all_matches:
        python_list_match = all_matches[-1]  # Get the last occurrence
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
        ui.print_warning("No variable names found in LLM response. This may indicate no relevant variables for this hypothesis.")
        # Return empty list instead of falling back to all variables
        # This allows the calling code to detect and handle the "no relevant variables" case
        return []
    
    # Remove duplicates while preserving order
    variable_names = list(dict.fromkeys(variable_names))
    
    ui.print_info(f":dart: Selected {len(variable_names)} variables: {variable_names}")
    return variable_names

def replace_unknown_values_with_nan(df: pd.DataFrame, data_cache: DataCache) -> pd.DataFrame:
    """
    Replace unknown/missing values with NaN across all columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        data_cache (DataCache): Data cache containing the JSON dictionary
    
    Returns:
        pd.DataFrame: Dataframe with unknown values replaced with NaN
    """
    ui.print_info(f"🔧 Replacing unknown values with NaN...")
    
    # Get the JSON dictionary from cache
    dictionary = data_cache.get_json_dictionary()
    
    processed_df = df.copy()
    total_replacements = 0
    
    # Keywords that indicate unknown/missing values
    unknown_keywords = ['unknown', 'not applicable', 'not administered', 'not available', 'refused', 'missing', 'n/a']
    
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
            
            # Find unknown codes (process all non-range entries, including when range is present)
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
                    # Convert code to appropriate type for comparison
                    try:
                        code_numeric = int(code)
                        if code_numeric in processed_df[col].values:
                            mask = processed_df[col] == code_numeric
                            count = mask.sum()
                            processed_df.loc[mask, col] = np.nan
                            col_replacements += count
                            ui.print_info(f"   📊 {col}: Replaced {count} instances of '{code}' with NaN")
                    except ValueError:
                        # If not numeric, compare as string
                        if code in processed_df[col].astype(str).values:
                            mask = processed_df[col].astype(str) == code
                            count = mask.sum()
                            processed_df.loc[mask, col] = np.nan
                            col_replacements += count
                            ui.print_info(f"   📊 {col}: Replaced {count} instances of '{code}' with NaN")
                
                total_replacements += col_replacements
                if col_replacements > 0:
                    ui.print_info(f"   ✅ {col}: Total replacements = {col_replacements}")
    
    ui.print_info(f"📊 Total unknown values replaced: {total_replacements}")
    return processed_df


def create_dummy_variables_for_categorical(df: pd.DataFrame, data_cache: DataCache) -> pd.DataFrame:
    """
    Create dummy variables for categorical columns (one per category, excluding unknown).
    
    Args:
        df (pd.DataFrame): Input dataframe
        data_cache (DataCache): Data cache containing the JSON dictionary
    
    Returns:
        pd.DataFrame: Dataframe with dummy variables created
    """
    ui.print_info(f"🔧 Creating dummy variables for categorical columns...")
    
    # Get the JSON dictionary from cache
    dictionary = data_cache.get_json_dictionary()
    
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
                    
                    # Clean up category names for dummy variable column names (replace spaces with underscores, make uppercase)
                    def clean_category_name(category_name):
                        if pd.isna(category_name):
                            return category_name
                        # Replace spaces with underscores, remove other special characters, make uppercase
                        cleaned = re.sub(r'\s+', '_', str(category_name))  # Replace spaces with underscores
                        cleaned = re.sub(r'[^\w_]', '', cleaned)  # Remove special characters except underscores
                        cleaned = cleaned.upper()  # Make uppercase
                        return cleaned
                    
                    col_for_dummies_cleaned = col_for_dummies.apply(clean_category_name)
                    
                    # Create dummy variables, excluding NaN values
                    dummies = pd.get_dummies(col_for_dummies_cleaned, prefix=col, dummy_na=False)
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


def get_column_descriptions(data_cache: DataCache, processed_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Get descriptions for all columns in the processed dataset from the JSON dictionary.
    
    Args:
        data_cache (DataCache): Data cache containing the JSON dictionary
        processed_df (Optional[pd.DataFrame]): Processed dataframe (if None, uses cached processed data)
    
    Returns:
        Dict[str, str]: Dictionary mapping column names to their descriptions
    """
    ui.print_info(f"🔍 Getting column descriptions from JSON dictionary...")
    
    # Use provided dataframe or get from cache
    if processed_df is not None:
        df = processed_df
    else:
        df = data_cache.get_processed_data()
        if df is None:
            raise ValueError("No processed data available in cache and no dataframe provided")
    
    # Get the JSON dictionary from cache
    dictionary = data_cache.get_json_dictionary()
    
    column_descriptions = {}
    
    for col in df.columns:
        description = "No description available"
        
        # Check if it's a dummy variable (contains underscore)
        if '_' in col:
            # Try to extract the original variable name from dummy variable
            # Dummy variables are typically named like "VAR_1", "VAR_2", etc.
            possible_base_vars = []
            for var_name in dictionary.keys():
                if col.startswith(var_name + '_'):
                    possible_base_vars.append(var_name)
            
            if possible_base_vars:
                # Use the longest matching base variable name
                base_var = max(possible_base_vars, key=len)
                var_info = dictionary[base_var]
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
                for var_name, var_info in dictionary.items():
                    if col.upper() == var_name.upper():
                        description = var_info.get('Description', 'No description available')
                        break
        else:
            # Regular column - try to find exact match
            for var_name, var_info in dictionary.items():
                if col.upper() == var_name.upper():
                    description = var_info.get('Description', 'No description available')
                    break
        
        column_descriptions[col] = description
        ui.print_info(f"   📝 {col}: {description}")
    
    ui.print_info(f"✅ Retrieved descriptions for {len(column_descriptions)} columns")
    return column_descriptions


def add_variable_descriptions_to_conversation(conversation_manager, data_cache: DataCache) -> None:
    """
    Add variable descriptions to a conversation manager as query context for the scientist agent.
    This can be called anytime to provide the scientist agent with variable context.
    
    Args:
        conversation_manager: Conversation manager to add the variable descriptions to
        data_cache (DataCache): Data cache containing the processed data and descriptions
    """
    ui.print_info(f"🔍 Adding variable descriptions to conversation context...")
    
    try:
        # Get column descriptions from cache
        column_descriptions = data_cache.get_processed_column_descriptions()
        
        # Create variable information string with descriptions
        variable_info_str = []
        for col, description in column_descriptions.items():
            variable_info_str.append(f"- {col}: {description}")
        
        variable_descriptions_str = "\n".join(variable_info_str)
        
        # Add to conversation manager
        variable_context_message = {
            "role": "System",
            "content": (
                f"VARIABLE CONTEXT FOR ANALYSIS:\n"
                f"Available variables in the processed dataset:\n"
                f"{variable_descriptions_str}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Use these exact variable names in all analysis queries\n"
                f"2. Refer to descriptions to understand variable meanings\n"
                f"3. CRITICAL: Always remove NaN values before performing any statistical tests or analyses\n"
                f"   Use .dropna() or .dropna(subset=[column_names]) to remove missing values for specific variables\n"
                f"4. Use exact column names for statistical tests and visualizations\n\n"
                f"Total variables available: {len(column_descriptions)}"
            )
        }
        
        conversation_manager.messages.append(variable_context_message)
        ui.print_success(f"✅ Added variable descriptions for {len(column_descriptions)} variables to conversation context")
        
    except Exception as e:
        ui.print_error(f"❌ Error adding variable descriptions to conversation: {str(e)}")


def show_processed_dataset_descriptions(data_cache: DataCache) -> None:
    """
    Standalone function to show column descriptions for a processed dataset.
    This can be called independently to review what variables are available.
    
    Args:
        data_cache (DataCache): Data cache containing the processed data and descriptions
    """
    ui.print_info(f"🔍 Showing descriptions for processed dataset from cache")
    
    try:
        # Get column descriptions from cache
        column_descriptions = data_cache.get_processed_column_descriptions()
        
        # Display formatted output
        ui.print_success(f"✅ Found {len(column_descriptions)} variables in processed dataset:")
        ui.print_info("=" * 80)
        
        for col, description in column_descriptions.items():
            ui.print_info(f"📊 {col}")
            ui.print_info(f"   📝 {description}")
            ui.print_info("-" * 40)
        
        ui.print_info("=" * 80)
        ui.print_success(f"✅ Displayed descriptions for {len(column_descriptions)} variables")
        
    except Exception as e:
        ui.print_error(f"❌ Error showing dataset descriptions: {str(e)}")


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


def simplified_data_preprocessing(model_client: ModelClient, hypothesis: str, data_cache: DataCache, 
                                 sample_size: int = None, seed: int = None, 
                                 output_path: str = None) -> pd.DataFrame:
    """
    Main function to perform simplified data preprocessing.
    
    Args:
        model_client (ModelClient): Model client for LLM interactions
        hypothesis (str): The hypothesis to analyze
        data_cache (DataCache): Data cache containing the original data
        output_path (str): Path where to save the processed data
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for data preprocessing operations. Please install pandas.")
    
    ui.print_info(f"🚀 Starting simplified data preprocessing...")
    ui.log_system_event("INFO", "Starting simplified data preprocessing...", "preprocessing")
    ui.print_info(f"🔬 Hypothesis: {hypothesis}")
    ui.log_system_event("INFO", f"Hypothesis: {hypothesis}", "preprocessing")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = "working/processed_dataset.csv"
    
    # Step 1: Get the original CSV data from cache (with optional sampling)
    ui.print_info(f"📖 Getting original CSV data from cache...")
    ui.log_preprocessing_step(1, "Data Loading", "Getting original CSV data from cache", "started")
    
    if sample_size is not None:
        ui.print_info(f"📊 Sampling {sample_size} rows from dataset...")
        df = data_cache.sample_dataset_rows(sample_size, seed)
        ui.log_preprocessing_step(1, "Data Loading", f"Sampled {sample_size} rows from dataset", "completed")
    else:
        df = data_cache.get_original_csv_data()
        ui.log_preprocessing_step(1, "Data Loading", f"Original data shape: {df.shape[0]} rows, {df.shape[1]} columns", "completed")
    
    ui.print_info(f"📊 Working data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Query LLM to filter columns
    ui.print_info(f"🤖 Step 1: Querying LLM for variable selection...")
    ui.log_preprocessing_step(2, "Variable Selection", "Querying LLM for variable selection", "started")
    selected_variables = query_llm_for_variable_selection(model_client, hypothesis, data_cache)
    
    # Check if no relevant variables were found
    if not selected_variables:
        ui.print_warning("❌ No relevant variables found for this hypothesis. Cannot proceed with analysis.")
        ui.log_preprocessing_step(2, "Variable Selection", "No relevant variables found", "failed")
        # Return a special indicator that no variables were found
        raise ValueError("NO_RELEVANT_VARIABLES")
    
    ui.log_preprocessing_step(2, "Variable Selection", f"Selected {len(selected_variables)} variables", "completed")
    
    # Filter dataframe to only include selected variables
    ui.log_preprocessing_step(3, "Data Filtering", "Filtering dataframe to selected variables", "started")
    available_columns = []
    missing_columns = []
    
    for var_name in selected_variables:
        # Ensure var_name is a string
        if not isinstance(var_name, str):
            ui.print_warning(f"⚠️ Skipping non-string variable name: {var_name} (type: {type(var_name)})")
            continue
            
        # Try exact match first
        if var_name in df.columns:
            available_columns.append(var_name)
        else:
            # Try case-insensitive match
            found = False
            for col in df.columns:
                if col.upper() == var_name.upper():
                    available_columns.append(col)
                    found = True
                    break
            if not found:
                missing_columns.append(var_name)
    
    if missing_columns:
        ui.print_warning(f"⚠️ Could not find {len(missing_columns)} selected columns in data: {missing_columns}")
    
    if available_columns:
        df = df[available_columns]
        ui.print_info(f"🎯 Filtered to {len(available_columns)} selected columns")
    else:
        ui.print_warning("⚠️ No selected columns found in data. Cannot proceed with analysis.")
        # Return a special indicator that no variables were found
        raise ValueError("NO_RELEVANT_VARIABLES")
    
    # Step 3: Replace unknown values with NaN
    ui.print_info(f"🔧 Step 2: Replacing unknown values with NaN...")
    df = replace_unknown_values_with_nan(df, data_cache)
    
    # Step 4: Create dummy variables for categorical columns
    ui.print_info(f"🔧 Step 3: Creating dummy variables for categorical columns...")
    df = create_dummy_variables_for_categorical(df, data_cache)
    
    # Step 5: Save processed data
    ui.print_info(f"💾 Step 4: Saving processed data...")
    save_processed_data(df, output_path)
    
    ui.print_success(f"✅ Simplified data preprocessing completed!")
    ui.print_info(f"📊 Final data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


# =============================================================================
# COMPATIBILITY WRAPPER CLASS
# =============================================================================

class SimplifiedDataProcessor:
    """Simplified data processor that maintains compatibility with the current pipeline."""
    
    def __init__(self, model_client: ModelClient, data_cache: Optional[DataCache] = None, config=None):
        """Initialize with model client and optional data cache."""
        self.model_client = model_client
        self.data_cache = data_cache
        self.config = config
    
    def run_simplified_preprocessing(self, hypothesis: str, dataset_folder: str = None, 
                                   previous_iteration_context: str = None, 
                                   conversation_manager=None, 
                                   previously_used_variables: List[str] = None,
                                   sample_size: int = None, seed: int = None):
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
            # Check if data cache is available
            if not self.data_cache or not self.data_cache.is_initialized():
                ui.print_warning("⚠️ No data cache available. Cannot proceed with preprocessing.")
                return {
                    'success': False,
                    'error': 'No data cache available'
                }
            
            # Run the simplified preprocessing using cached data
            working_dir = self.config.working_dir if self.config else "working"
            output_path = f"{working_dir}/processed_dataset.csv"
            try:
                processed_df = simplified_data_preprocessing(
                    model_client=self.model_client,
                    hypothesis=hypothesis,
                    data_cache=self.data_cache,
                    sample_size=sample_size,
                    seed=seed,
                    output_path=output_path
                )
            except ValueError as e:
                if str(e) == "NO_RELEVANT_VARIABLES":
                    ui.print_warning("❌ No relevant variables found for this hypothesis. Marking as not applicable.")
                    return {
                        'success': False,
                        'error': 'No relevant variables found',
                        'no_relevant_variables': True
                    }
                else:
                    raise  # Re-raise other ValueError exceptions
            
            # Store processed data in cache
            self.data_cache.set_processed_data(processed_df)
            
            # Get statistics for the processed dataset from cache
            filtered_csv_statistics = self.data_cache.get_processed_dataset_statistics()
            
            # Get column descriptions from cache
            column_descriptions = self.data_cache.get_processed_column_descriptions()
            
            # Create variable information string with descriptions
            variable_info_str = []
            for col, description in column_descriptions.items():
                variable_info_str.append(f"- {col}: {description}")
            
            variable_descriptions_str = "\n".join(variable_info_str)
            
            # Add system message to conversation if conversation_manager is provided
            if conversation_manager:
                summarized_messages = [{
                    "role": "System",
                    "content": (
                        f"DATA PROCESSING COMPLETED\n"
                        f"Processed data saved to: {output_path}\n\n"
                        f"AVAILABLE VARIABLES FOR ANALYSIS:\n"
                        f"{variable_descriptions_str}\n\n"
                        f"DATASET STATISTICS:\n"
                        f"- Total rows: {filtered_csv_statistics.get('total_rows', 'N/A'):,}\n"
                        f"- Total columns: {filtered_csv_statistics.get('total_cols', 'N/A')}\n"
                        f"- Missing values: {filtered_csv_statistics.get('missing_data', 'N/A'):,} ({filtered_csv_statistics.get('missing_percentage', 'N/A')}%)\n"
                        f"- Numeric columns: {filtered_csv_statistics.get('numeric_cols', 'N/A')}\n"
                        f"- Categorical columns: {filtered_csv_statistics.get('categorical_cols', 'N/A')}\n\n"
                        f"IMPORTANT INSTRUCTIONS:\n"
                        f"1. Use the variable names exactly as shown above for all analysis queries\n"
                        f"2. Refer to the descriptions to understand what each variable represents\n"
                        f"3. CRITICAL: Always remove NaN values before performing any statistical tests or analyses\n"
                        f"   Use .dropna() or .dropna(subset=[column_names]) to remove missing values for specific variables\n"
                        f"4. When creating visualizations or statistical tests, use the exact column names from the dataset\n\n"
                        f"QUERY CONTEXT: The scientist agent now has access to {len(column_descriptions)} variables with their descriptions for hypothesis testing and analysis."
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
                'json_dictionary_path': self.data_cache.get_json_path(),  # Include the JSON dictionary path
                'preprocessed_data': {
                    'dataframe': processed_df,
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
    def get_dataset_statistics(self, csv_path: str = None) -> Dict[str, Any]:
        """Wrapper for dataset statistics."""
        if self.data_cache and self.data_cache.is_initialized():
            return self.data_cache.get_dataset_statistics()
        else:
            return get_dataset_statistics(csv_path)
    
    def detect_dataset_files(self, folder_path: str) -> Tuple[str, str]:
        """Wrapper for file detection."""
        return detect_dataset_files(folder_path)
    
    def set_data_cache(self, data_cache: DataCache) -> None:
        """Set the data cache for this processor."""
        self.data_cache = data_cache


# Example usage functions
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


def show_dataset_descriptions_example():
    """
    Example function showing how to display column descriptions for a processed dataset.
    """
    # Show descriptions for the default processed dataset
    show_processed_dataset_descriptions()
    
    # Or specify custom paths
    # show_processed_dataset_descriptions(
    #     csv_path="working/processed_dataset.csv",
    #     json_path="path/to/your/dictionary.json"
    # )


def add_descriptions_to_conversation_example():
    """
    Example function showing how to add variable descriptions to a conversation manager.
    """
    # This would be called when you want to provide variable context to the scientist agent
    # conversation_manager = your_conversation_manager_instance
    
    # Add descriptions to conversation context
    # add_variable_descriptions_to_conversation(conversation_manager)
    
    # Or specify custom paths
    # add_variable_descriptions_to_conversation(
    #     conversation_manager,
    #     csv_path="working/processed_dataset.csv",
    #     json_path="path/to/your/dictionary.json"
    # )
