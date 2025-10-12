# Integration Guide: Simplified Data Processing with Current Pipeline

This guide shows how to integrate the new simplified data processing with your existing DiscoAD_MAS pipeline.

## Current Pipeline Structure

Your current pipeline works as follows:
1. **discover.py** → Main entry point
2. **DiscoveryAgentSystem** → Main orchestrator
3. **ExperimentRunner** → Handles experiment execution
4. **DataProcessor** → Current data processing (complex, 1300+ lines)
5. **run_manual_preprocessing_conversation()** → Current preprocessing method

## Integration Options

### Option 1: Replace Current DataProcessor (Recommended)

Replace the current complex `DataProcessor` with the simplified version.

#### Step 1: Update the import in experiment_runner.py

```python
# In experiment_runner.py, line 17, change:
from .data_processing import DataProcessor

# To:
from .simplified_data_processing import SimplifiedDataProcessor
```

#### Step 2: Update the DataProcessor initialization

```python
# In experiment_runner.py, line 32, change:
self.data_processor = DataProcessor(model_client)

# To:
self.data_processor = SimplifiedDataProcessor(model_client)
```

#### Step 3: Update the preprocessing call

```python
# In experiment_runner.py, around line 241, change:
preprocessing_result = self.data_processor.run_manual_preprocessing_conversation(
    hypothesis, self.config.dataset_folder, previous_conversation_summary, 
    self.conversation_manager, previously_used_variables
)

# To:
preprocessing_result = self.data_processor.run_simplified_preprocessing(
    hypothesis, self.config.dataset_folder, previous_conversation_summary, 
    self.conversation_manager, previously_used_variables
)
```

### Option 2: Create a Wrapper Class (Safer)

Create a wrapper that maintains the same interface but uses the simplified processing internally.

## Implementation: SimplifiedDataProcessor Class

Here's the wrapper class that maintains compatibility with your current pipeline:

```python
# Add this to simplified_data_processing.py

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
```

## Quick Integration Steps

### Step 1: Add the wrapper class to simplified_data_processing.py

Add the `SimplifiedDataProcessor` class to the end of your `simplified_data_processing.py` file.

### Step 2: Update experiment_runner.py

```python
# Change the import (line 17):
from .simplified_data_processing import SimplifiedDataProcessor

# Change the initialization (line 32):
self.data_processor = SimplifiedDataProcessor(model_client)

# Change the method call (around line 241):
preprocessing_result = self.data_processor.run_simplified_preprocessing(
    hypothesis, self.config.dataset_folder, previous_conversation_summary, 
    self.conversation_manager, previously_used_variables
)
```

### Step 3: Update __init__.py (optional)

```python
# In discovery_agent_pkg/__init__.py, add:
from .simplified_data_processing import SimplifiedDataProcessor

# Add to __all__ list:
"SimplifiedDataProcessor"
```

## Benefits of the Simplified Version

1. **Cleaner Code**: 472 lines vs 1300+ lines
2. **Focused Functionality**: Does exactly what you requested
3. **Better Performance**: Streamlined processing
4. **Easier Maintenance**: Simpler logic flow
5. **Same Interface**: Drop-in replacement

## Testing the Integration

1. **Backup your current data_processing.py**:
   ```bash
   cp discovery_agent_pkg/data_processing.py discovery_agent_pkg/data_processing_backup.py
   ```

2. **Make the changes** as described above

3. **Test with a simple hypothesis**:
   ```bash
   python discover.py --hypothesis-mode "direct hypothesis" --hypothesis "Age affects memory" --dataset-folder "/path/to/your/dataset"
   ```

4. **Check the output** in the `working/` directory for the processed dataset

## Rollback Plan

If you need to rollback:
1. Restore the original import: `from .data_processing import DataProcessor`
2. Restore the original initialization: `self.data_processor = DataProcessor(model_client)`
3. Restore the original method call: `run_manual_preprocessing_conversation`

## Configuration

The simplified version uses the same configuration as your current pipeline:
- `self.config.dataset_folder` for dataset location
- Same output directory structure (`working/`)
- Same conversation manager integration
- Same error handling and logging

This integration maintains full compatibility with your existing pipeline while providing the simplified, focused data processing you requested.
