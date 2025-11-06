"""
Experiment runner module for the Discovery Agent.

This module handles all experiment-related operations including:
- Sampling run orchestration
- Hypothesis processing and iteration management
- Experiment statistics and reporting
- Outer loop iteration management
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from ..ui.interface import ui
from .conversation import ConversationManager
from ..execution.executor import CodeExecutor
from ..agents.models import ModelClient
from ..agents.scientist import create_scientist
from ..agents.critic import create_critic
from ..agents.planner import create_enhanced_planning_agent
from ..data.processor import SimplifiedDataProcessor, add_variable_descriptions_to_conversation
from ..data.cache import DataCache
from .hypothesis import HypothesisManager
from ..data.file_manager import FileManager
from ..execution.utils import execute_code_from_response
from ..utils.xml_parser import parse_critic_judgment


class ExperimentRunner:
    """Handles all experiment execution operations for the Discovery Agent."""
    
    def __init__(self, model_client: ModelClient, config=None, file_manager=None):
        """Initialize the experiment runner with all required components."""
        self.config = config
        self.model_client = model_client
        
        # Initialize data cache
        self.data_cache = DataCache()
        
        # Initialize all components
        self.data_processor = SimplifiedDataProcessor(model_client, self.data_cache, config)
        self.hypothesis_manager = HypothesisManager(model_client)
        # Use provided file manager or create a new one
        self.file_manager = file_manager if file_manager is not None else FileManager(config)
        
        # Initialize agents
        self.scientist = create_scientist(model_client, config)
        self.critic = create_critic(model_client)
        
        # Initialize planning agent only if enabled in config (default: enabled)
        if config is None or getattr(config, 'enable_planning_agent', True):
            self.enhanced_planning_agent = create_enhanced_planning_agent(model_client)
        else:
            self.enhanced_planning_agent = None
        
        # Initialize conversation and code execution
        self.conversation_manager = ConversationManager(model_client, config)
        self.code_executor = CodeExecutor(timeout_seconds=config.code_timeout if config else 30)
    
    def initialize_data_cache(self, dataset_folder: str) -> bool:
        """
        Initialize the data cache with CSV and JSON data from the dataset folder.
        
        Args:
            dataset_folder (str): Path to folder containing CSV and JSON files
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        return self.data_cache.initialize(dataset_folder)
    
    def initialize_sampling_run(self) -> Tuple[Optional[List[str]], Optional[int], Optional[bool], int, int]:
        """
        Initialize the sampling run by loading keywords and setting up tracking.
        
        Returns:
            Tuple containing: keywords, num_runs, verbose, total_hypotheses, total_approved
        """
        verbose = self.config.verbose
        
        # Check if using direct hypothesis mode
        if self.config.hypothesis_mode == "direct hypothesis":
            if not self.config.hypothesis:
                ui.print_error("❌ Direct hypothesis mode specified but no hypothesis provided.", "initialization")
                return None, None, None, 0, 0
            
            ui.print_info(f"🎯 Using direct hypothesis mode with provided hypothesis", verbose=verbose)
            # For direct hypothesis, we only need 1 run
            return None, 1, verbose, 0, 0
        
        # Check if using CSV batch mode
        if self.config.hypothesis_mode == "csv_batch":
            if not self.config.csv_file:
                ui.print_error("❌ CSV batch mode specified but no CSV file provided. Use --csv-file parameter.", "initialization")
                return None, None, None, 0, 0
            
            ui.print_info(f"📊 Using CSV batch mode with file: {self.config.csv_file}", verbose=verbose)
            # Load hypotheses from CSV to determine number of runs
            hypotheses_data = self.hypothesis_manager.load_hypotheses_from_csv(self.config.csv_file)
            if not hypotheses_data:
                ui.print_error("❌ No hypotheses found in CSV file.", "initialization")
                return None, None, None, 0, 0
            
            # Get sample_size for dataset sampling (not hypothesis sampling)
            sample_size = getattr(self.config, 'sample_size', None)
            
            if sample_size is not None:
                ui.print_info(f"📊 Dataset sampling enabled with sample_size: {sample_size}", verbose=verbose)
                # Store sample_size for use in data preprocessing
                self.dataset_sample_size = sample_size
            else:
                ui.print_info(f"📊 Using full dataset (no sampling)", verbose=verbose)
                self.dataset_sample_size = None
            
            # Determine number of runs based on hypotheses (not affected by dataset sampling)
            if self.config.start_index is not None or self.config.end_index is not None:
                # Use start_index and end_index mode for sequential processing
                start_index = getattr(self.config, 'start_index', None)
                end_index = getattr(self.config, 'end_index', None)
                
                # Set defaults for null values
                if start_index is None:
                    start_index = 0
                if end_index is None:
                    end_index = len(hypotheses_data) - 1
                
                # Validate start index
                if start_index < 0 or start_index >= len(hypotheses_data):
                    ui.print_error(f"❌ Start index {start_index} is invalid. Must be between 0 and {len(hypotheses_data)-1}.", "initialization")
                    return None, None, None, 0, 0
                
                # Validate end index
                if end_index < start_index or end_index >= len(hypotheses_data):
                    ui.print_error(f"❌ End index {end_index} is invalid. Must be between {start_index} and {len(hypotheses_data)-1} (inclusive).", "initialization")
                    return None, None, None, 0, 0
                
                # Calculate available range size
                available_range_size = end_index - start_index + 1
                num_runs = available_range_size
                ui.print_info(f"📚 Found {len(hypotheses_data)} hypotheses in CSV file, processing range {start_index} to {end_index} ({num_runs} runs)", verbose=verbose)
                self.sampled_csv_indices = None  # No sampling
            else:
                # Use all hypotheses if no start/end index specified
                num_runs = len(hypotheses_data)
                ui.print_info(f"📚 Found {len(hypotheses_data)} hypotheses in CSV file, processing all hypotheses ({num_runs} runs)", verbose=verbose)
                self.sampled_csv_indices = None  # No sampling
            
            return None, num_runs, verbose, 0, 0
        
        # Load ADRD keywords for generate/direct modes
        keywords = self.hypothesis_manager.load_adrd_keywords(self.config.keywords_file)
        if not keywords:
            ui.print_warning(f"🚫 No ADRD keywords found. Exiting.")
            return None, None, None, 0, 0
        
        ui.print_info(f"📚 Loaded {len(keywords)} ADRD keywords", verbose=verbose)
        
        # Get number of sampling runs from config or use default
        num_runs = getattr(self.config, 'num_sampling_runs', 10)
        ui.print_info(f"🎯 Running {num_runs} keyword sampling iterations", verbose=verbose)
        
        # Track results
        total_hypotheses = 0
        total_approved = 0
        
        return keywords, num_runs, verbose, total_hypotheses, total_approved
    
    def run_sampling_iterations(self, keywords: Optional[List[str]], num_runs: int, verbose: bool, 
                               total_hypotheses: int, total_approved: int) -> Tuple[int, int]:
        """
        Run the main sampling iterations loop.
        
        Args:
            keywords (Optional[List[str]]): List of available keywords (None for direct hypothesis mode)
            num_runs (int): Number of sampling runs to perform
            verbose (bool): Whether to use verbose output
            total_hypotheses (int): Current total hypotheses count
            total_approved (int): Current total approved count
        
        Returns:
            Tuple[int, int]: Updated total_hypotheses and total_approved counts
        """
        for run_idx in range(1, num_runs + 1):
            try:
                ui.print_separator(verbose=verbose)
                
                # Handle different modes
                if keywords is None:  # Direct hypothesis or CSV batch mode
                    if self.config.hypothesis_mode == "direct hypothesis":
                        ui.print_info(f"🎯 Direct Hypothesis Run {run_idx}/{num_runs}", verbose=verbose)
                        sampled_keywords = []  # No keywords needed for direct hypothesis
                    elif self.config.hypothesis_mode == "csv_batch":
                        ui.print_info(f"📊 CSV Batch Run {run_idx}/{num_runs}", verbose=verbose)
                        sampled_keywords = []  # No keywords needed for CSV batch mode
                    else:
                        ui.print_info(f"🎯 Direct Hypothesis Run {run_idx}/{num_runs}", verbose=verbose)
                        sampled_keywords = []  # No keywords needed for direct hypothesis
                else:
                    ui.print_info(f"🎲 Sampling Run {run_idx}/{num_runs}", verbose=verbose)
                    # Sample keywords with seed for deterministic results
                    sampled_keywords = self.hypothesis_manager.sample_keywords(keywords, self.config.num_keywords, seed=self.config.seed)
                    ui.print_info(f"🎯 Sampled keywords: {', '.join(sampled_keywords)}", verbose=verbose)
                
                # Create hypothesis from sampled keywords (for non-direct modes)
                if keywords is not None:
                    hypothesis_input = f"{', '.join(sampled_keywords)}"
                    ui.print_info(f"📝 Generated hypothesis: {hypothesis_input}", verbose=verbose)
                
                # Get hypothesis mode from config
                hypothesis_mode = self.config.hypothesis_mode
                
                if hypothesis_mode == "generate":
                    # Generate multiple hypotheses from sampled keywords
                    ui.print_info(f"🔬 Generating multiple hypotheses from sampled keywords...", verbose=verbose)
                    
                    # Get dataset dictionary and CSV statistics from cache if available
                    dataset_dictionary = None
                    csv_statistics = None
                    if self.data_cache.is_initialized():
                        try:
                            # Get the JSON dictionary from cache
                            dataset_dictionary = self.data_cache.get_json_dictionary()
                            
                            # Get CSV statistics from cache
                            csv_statistics = self.data_cache.get_dataset_statistics()
                            
                            ui.print_info(f"📚 Loaded dataset dictionary with {len(dataset_dictionary)} variables from cache", verbose=verbose)
                            if 'error' not in csv_statistics:
                                ui.print_info(f"📊 CSV statistics: {csv_statistics['total_rows']:,} rows, {csv_statistics['total_cols']} cols, {csv_statistics['missing_percentage']}% missing", verbose=verbose)
                            else:
                                ui.print_warning(f"⚠️ Could not get CSV statistics from cache: {csv_statistics['error']}", verbose=verbose)
                        except Exception as e:
                            ui.print_warning(f"⚠️ Could not get dataset dictionary or CSV from cache: {str(e)}", verbose=verbose)
                            dataset_dictionary = None
                            csv_statistics = None
                    
                    # Generate hypotheses with dataset information
                    hypotheses = self.hypothesis_manager.generate_hypotheses_from_keywords(
                        sampled_keywords, dataset_dictionary, csv_statistics
                    )
                    ui.print_hypothesis_list(hypotheses, generated=True, verbose=verbose)
                elif hypothesis_mode == "direct hypothesis":
                    # Use the provided direct hypothesis
                    if not self.config.hypothesis:
                        ui.print_error("❌ Direct hypothesis mode specified but no hypothesis provided. Please use --hypothesis parameter.", "hypothesis_mode")
                        continue
                    
                    ui.print_info(f"📝 Using provided direct hypothesis: {self.config.hypothesis}", verbose=verbose)
                    hypotheses = self.hypothesis_manager.set_direct_hypothesis(self.config.hypothesis)
                    ui.print_hypothesis_list(hypotheses, generated=False, verbose=verbose)
                elif hypothesis_mode == "csv_batch":
                    # Load hypothesis from CSV file for this run
                    ui.print_info(f"📊 Loading hypothesis from CSV file for run {run_idx}...", verbose=verbose)
                    hypotheses_data = self.hypothesis_manager.load_hypotheses_from_csv(self.config.csv_file)
                    if not hypotheses_data:
                        ui.print_error(f"❌ No hypotheses found in CSV file.", "csv_batch")
                        continue
                    
                    # Determine which hypothesis to use for this run
                    if hasattr(self, 'sampled_csv_indices') and self.sampled_csv_indices is not None:
                        # Use pre-sampled indices (sample_size mode)
                        csv_row_index = self.sampled_csv_indices[run_idx - 1] if run_idx <= len(self.sampled_csv_indices) else None
                        
                        # Check if we have enough data
                        if csv_row_index is None or csv_row_index >= len(hypotheses_data):
                            ui.print_error(f"❌ No hypothesis found for run {run_idx} (CSV row {csv_row_index}) in CSV file.", "csv_batch")
                            continue
                    else:
                        # Original sequential behavior using start_index and end_index
                        start_index = getattr(self.config, 'start_index', None)
                        end_index = getattr(self.config, 'end_index', None)
                        
                        # Set defaults for null values
                        if start_index is None:
                            start_index = 0
                        if end_index is None:
                            end_index = len(hypotheses_data) - 1
                        
                        csv_row_index = start_index + run_idx - 1
                        
                        # Check if we have enough data and within range
                        if csv_row_index > end_index or csv_row_index >= len(hypotheses_data):
                            ui.print_error(f"❌ No hypothesis found for run {run_idx} (CSV row {csv_row_index}) in CSV file. Range is {start_index} to {end_index}.", "csv_batch")
                            continue
                    
                    # Get the hypothesis for this run (already 0-based index)
                    hypothesis_data = hypotheses_data[csv_row_index]
                    hypothesis_text = hypothesis_data['hypothesis']
                    hypothesis_index = hypothesis_data['index']
                    
                    ui.print_info(f"📝 Using hypothesis from CSV (row {hypothesis_index}): {hypothesis_text}", verbose=verbose)
                    hypotheses = self.hypothesis_manager.set_direct_hypothesis(hypothesis_text)
                    ui.print_hypothesis_list(hypotheses, generated=False, verbose=verbose)
                    
                    # Note: CSV status updates removed to preserve original batch.csv
                else:  # "direct" mode
                    # Use sampled keywords directly as a single hypothesis
                    ui.print_info(f"📝 Using sampled keywords directly as hypothesis...", verbose=verbose)
                    hypotheses = self.hypothesis_manager.create_single_hypothesis_from_keywords(sampled_keywords)
                    ui.print_hypothesis_list(hypotheses, generated=False, verbose=verbose)
                
                # Process each generated hypothesis
                run_approved_count = 0
                for idx, hypothesis in enumerate(hypotheses, 1):
                    try:
                        status = self.process_hypothesis(hypothesis, idx, run_idx, sampled_keywords, verbose)
                        if status == "HYPOTHESIS VERIFIED":
                            run_approved_count += 1
                    except Exception as e:
                        # Handle unhandled exceptions in hypothesis processing
                        ui.print_error(f"❌ Hypothesis {idx} failed with exception: {str(e)}", "hypothesis_processing")
                        ui.print_info(f"🔄 Continuing with next hypothesis...", verbose=verbose)
                        
                        # Note: CSV status updates removed to preserve original batch.csv
                        
                        # Continue to next hypothesis instead of terminating the job
                        continue
                
                # Print run summary
                run_approval_rate = (run_approved_count / len(hypotheses)) if hypotheses else 0
                ui.print_info(f"📊 Run {run_idx} Summary:", verbose=verbose)
                ui.print_info(f"   🎯 Keywords: {', '.join(sampled_keywords)}", verbose=verbose)
                ui.print_info(f"   📝 Generated hypotheses: {len(hypotheses)}", verbose=verbose)
                ui.print_info(f"   ✅ Approved hypotheses: {run_approved_count}", verbose=verbose)
                ui.print_info(f"   📈 Approval rate: {run_approval_rate:.2%}", verbose=verbose)
                
                # Update totals
                total_hypotheses += len(hypotheses)
                total_approved += run_approved_count
                
            except Exception as e:
                # Handle unhandled exceptions in run processing
                ui.print_error(f"❌ Run {run_idx} failed with exception: {str(e)}", "run_processing")
                ui.print_info(f"🔄 Continuing with next run...", verbose=verbose)
                
                # Note: CSV status updates removed to preserve original batch.csv
                
                # Continue to next run instead of terminating the job
                continue
        
        return total_hypotheses, total_approved
    
    def process_hypothesis(self, hypothesis: str, idx: int, run_idx: int, 
                          sampled_keywords: List[str], verbose: bool) -> str:
        """
        Process a single hypothesis through the outer loop: data prep -> planner -> scientist/critic -> if not approved, go back to data prep with different variables.
        
        Args:
            hypothesis (str): The hypothesis to process
            idx (int): Index of the hypothesis
            run_idx (int): Run index
            sampled_keywords (List[str]): Keywords used for this hypothesis
            verbose (bool): Whether to use verbose output
        
        Returns:
            str: Status of hypothesis ("HYPOTHESIS VERIFIED", "HYPOTHESIS NOT VERIFIED", "NOT APPLICABLE")
        """
        import time
        start_time = time.time()
        
        ui.print_progress(idx, len(self.hypothesis_manager.get_hypotheses()), verbose=verbose)
        ui.print_separator(verbose=verbose)
        ui.print_info(f"🔍 Working on hypothesis {idx}: {hypothesis}", verbose=verbose)
        
        # Start JSON logging for this hypothesis
        ui.start_hypothesis_logging(idx, hypothesis, sampled_keywords)
        
        # Track outer loop iterations
        outer_loop_iterations = self.config.outer_loop_iterations
        approved = None
        num_turns_to_approve = None
        num_turns_to_autoapprove = self.config.auto_approve_after_attempts
        previous_conversation_summary = None
        previously_used_variables = []  # Track variables used in previous iterations
        total_outer_iterations = 0
        final_iteration_turns = None
        
        for outer_iteration in range(1, outer_loop_iterations + 1):
            ui.print_separator(verbose=verbose)
            ui.print_info(f"🔄 OUTER LOOP ITERATION {outer_iteration}/{outer_loop_iterations}", verbose=verbose)
            
            # Start JSON logging for this outer iteration
            ui.start_outer_iteration_logging(outer_iteration)
            
            # Reset conversation for each outer loop iteration
            self.conversation_manager.reset_conversation()
            
            # Add previous conversation summary as system message if available (at index 0)
            if previous_conversation_summary and outer_iteration > 1:
                ui.print_info(f"📝 Adding previous conversation summary as context...", verbose=verbose)
                self.conversation_manager.messages.append({
                    "role": "System", 
                    "content": f"PREVIOUS ITERATION SUMMARY:\n{previous_conversation_summary}\n\nThis is iteration {outer_iteration} of {outer_loop_iterations}. Use the above summary to avoid repeating the same approaches and try different variables or methods."
                })
            
            # Add user input (will be at index 0 if no summary, or index 1 if summary exists)
            self.conversation_manager.add_user_input(hypothesis)
            
            # Run manual preprocessing for this specific hypothesis and iteration
            ui.print_info(f"🔧 Running manual preprocessing for hypothesis {idx}, iteration {outer_iteration}...", verbose=verbose)
            if previously_used_variables:
                ui.print_info(f"📋 Passing {len(previously_used_variables)} previously used variables to avoid repetition", verbose=verbose)
            
            try:
                # Run preprocessing using cached data
                preprocessing_result = self.data_processor.run_simplified_preprocessing(
                    hypothesis, None, previous_conversation_summary, self.conversation_manager, 
                    previously_used_variables, self.dataset_sample_size, self.config.seed
                )
                if not preprocessing_result.get('success', False):
                    # Check if this is a "no relevant variables" case
                    if preprocessing_result.get('no_relevant_variables', False):
                        ui.print_warning(f"❌ No relevant variables found for hypothesis {idx}. Marking as not applicable.", verbose=verbose)
                        # Mark hypothesis as not verified and return early
                        return self._mark_hypothesis_not_verified(hypothesis, idx, run_idx, verbose)
                    else:
                        ui.print_warning(f"⚠️ Manual preprocessing failed for hypothesis {idx}, iteration {outer_iteration}, but continuing with analysis...", verbose=verbose)
                        # Set default values for failed preprocessing
                        filtered_csv_statistics = None
                        relevant_variables = None
                        json_dictionary_path = None
                else:
                    # Extract information for enhanced planning
                    filtered_csv_statistics = preprocessing_result.get('filtered_csv_statistics')
                    relevant_variables = preprocessing_result.get('relevant_variables')
                    json_dictionary_path = preprocessing_result.get('json_dictionary_path')
                    
                    # Track the variables used in this iteration for next iteration
                    if relevant_variables:
                        current_iteration_variables = list(relevant_variables.keys())
                        previously_used_variables.extend(current_iteration_variables)
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_variables = []
                        for var in previously_used_variables:
                            if var not in seen:
                                seen.add(var)
                                unique_variables.append(var)
                        previously_used_variables = unique_variables
                        ui.print_info(f"📋 Updated previously used variables list: {len(previously_used_variables)} total variables", verbose=verbose)
            except Exception as e:
                ui.print_error(f"Manual preprocessing failed for hypothesis {idx}, iteration {outer_iteration}: {str(e)}", "manual_preprocessing", verbose=verbose)
                # Set default values for failed preprocessing
                filtered_csv_statistics = None
                relevant_variables = None
                json_dictionary_path = None
                # Continue with the loop even if preprocessing fails
            
            # Run conversation until approved
            iteration_approved, iteration_turns_to_approve, iteration_turns_to_autoapprove = self.run_conversation_until_approved(
                verbose, 
                hypothesis, 
                filtered_csv_statistics, 
                relevant_variables,
                json_dictionary_path
            )
            
            # Track this iteration
            total_outer_iterations = outer_iteration
            final_iteration_turns = iteration_turns_to_approve if iteration_approved == "HYPOTHESIS VERIFIED" else iteration_turns_to_autoapprove
            
            # Display iteration result
            if iteration_approved == "HYPOTHESIS VERIFIED":
                result_display = "✅ VERIFIED"
            elif iteration_approved == "NOT APPLICABLE":
                result_display = "⚠️ NOT APPLICABLE"
            else:
                result_display = "❌ NOT VERIFIED"
            ui.print_info(f"📊 Iteration {outer_iteration} result: {result_display}", verbose=verbose)
            
            # End JSON logging for this outer iteration
            ui.end_outer_iteration_logging(iteration_approved)
            
            if iteration_approved == "HYPOTHESIS VERIFIED":
                # If approved, we're done with this hypothesis
                approved = iteration_approved
                num_turns_to_approve = iteration_turns_to_approve
                ui.print_success(f"🎉 Hypothesis {idx} VERIFIED in iteration {outer_iteration}!", verbose=verbose)
                break
            elif iteration_approved == "NOT APPLICABLE":
                # If not applicable, we're done with this hypothesis
                approved = iteration_approved
                ui.print_warning(f"⚠️ Hypothesis {idx} NOT APPLICABLE in iteration {outer_iteration}!", verbose=verbose)
                break
            else:
                # If not approved, create summary for next iteration
                if outer_iteration < outer_loop_iterations:
                    ui.print_info(f"📝 Creating conversation summary for next iteration...", verbose=verbose)
                    previous_conversation_summary = self.conversation_manager.create_outer_loop_summary()
                    ui.print_info(f"🔄 Will try different approach in iteration {outer_iteration + 1}...", verbose=verbose)
                else:
                    ui.print_warning(f"⚠️ Hypothesis {idx} NOT VERIFIED after {outer_loop_iterations} iterations", verbose=verbose)
        
        # If no decision was made, default to not verified
        if approved is None:
            approved = "HYPOTHESIS NOT VERIFIED"
        
        ui.print_approval_status(approved, verbose=verbose)
        
        # End JSON logging for this hypothesis
        ui.end_hypothesis_logging(approved)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Write CSV entry for this hypothesis (only once, after all outer loop iterations are complete)
        # For CSV batch mode, use category as keywords if available
        tracking_keywords = sampled_keywords
        if self.config.hypothesis_mode == "csv_batch":
            # Try to get category from CSV data for better tracking
            try:
                hypotheses_data = self.hypothesis_manager.load_hypotheses_from_csv(self.config.csv_file)
                if hypotheses_data and run_idx <= len(hypotheses_data):
                    hypothesis_data = hypotheses_data[run_idx - 1]
                    if 'Category' in hypothesis_data:
                        tracking_keywords = [hypothesis_data['Category']]
            except:
                pass  # Fall back to empty list if there's an issue
        
        self.file_manager.write_csv_entry(
            run_number=run_idx,
            keywords=tracking_keywords,
            hypothesis_index=idx,
            hypothesis_text=hypothesis,
            outer_iterations_used=total_outer_iterations,  # This will be the final iteration number
            inner_iterations_used=final_iteration_turns,
            hypothesis_approved=approved,
            turns_to_approve=num_turns_to_approve,
            execution_time_seconds=execution_time
        )
        
        # Note: CSV batch mode no longer updates the source CSV file to preserve original data
        
        # Clean up working directory after each hypothesis (approved or not)
        ui.print_info(f"🧹 Cleaning up working directory after hypothesis {idx}...", verbose=verbose)
        self.file_manager.cleanup_working_directory(recreate=False)
        
        return approved
    
    def _mark_hypothesis_not_verified(self, hypothesis: str, idx: int, run_idx: int, verbose: bool) -> str:
        """
        Mark a hypothesis as not applicable when no relevant variables are found.
        
        Args:
            hypothesis (str): The hypothesis text
            idx (int): Hypothesis index
            run_idx (int): Run index
            verbose (bool): Whether to print verbose output
            
        Returns:
            str: Always returns "NOT APPLICABLE"
        """
        import time
        start_time = time.time()
        
        ui.print_info(f"📝 Marking hypothesis {idx} as not applicable due to no relevant variables...", verbose=verbose)
        
        # Calculate execution time (minimal time for this case)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Record in CSV tracking file
        self.file_manager.write_csv_entry(
            run_number=run_idx,
            keywords=[],  # No keywords for not applicable
            hypothesis_index=idx,
            hypothesis_text=hypothesis,
            outer_iterations_used=0,  # No iterations used
            inner_iterations_used=0,  # No iterations used
            hypothesis_approved="NOT APPLICABLE",
            turns_to_approve=None,
            execution_time_seconds=execution_time
        )
        
        # Note: CSV status updates removed to preserve original batch.csv
        
        # Clean up working directory
        ui.print_info(f"🧹 Cleaning up working directory after marking hypothesis {idx} as not applicable...", verbose=verbose)
        self.file_manager.cleanup_working_directory(recreate=False)
        
        return "NOT APPLICABLE"
    
    def run_conversation_until_approved(self, verbose: bool, hypothesis: str = None, 
                                      filtered_csv_statistics: Dict[str, Any] = None, 
                                      relevant_variables: Dict[str, str] = None,
                                      json_dictionary_path: str = None) -> Tuple[str, Optional[int], int]:
        """
        Run the conversation loop until a hypothesis is approved or max attempts reached.
        
        Args:
            verbose (bool): Whether to use verbose output
            hypothesis (str): The hypothesis being tested
            filtered_csv_statistics (Dict[str, Any]): Statistics from filtered dataset
            relevant_variables (Dict[str, str]): Relevant variables for the hypothesis
            json_dictionary_path (str): Path to the JSON dictionary file
        
        Returns:
            Tuple[str, Optional[int], int]: (status, turns_to_approve, turns_to_autoapprove)
        """
        approved = None  # Start with None to indicate no decision yet
        num_turns_to_approve = None
        num_turns_to_autoapprove = self.config.auto_approve_after_attempts
                
        # Planning agent round (only on first turn and if enabled)
        if self.conversation_manager.get_turns() == 0 and self.enhanced_planning_agent is not None:
            ui.print_info("📋 Starting Planning Agent round...", verbose=verbose)
            self.planning_agent_round(hypothesis, filtered_csv_statistics, verbose=verbose)
        elif self.conversation_manager.get_turns() == 0 and self.enhanced_planning_agent is None:
            ui.print_info("📋 Planning Agent is disabled, skipping planning round...", verbose=verbose)

        while approved is None:
            try:
                ui.print_info(f"🔄 Conversation turn {self.conversation_manager.get_turns() + 1}", verbose=verbose)
                
                if self.conversation_manager.should_auto_approve():
                    ui.print_warning(f"⚠️ Maximum attempts reached. Results not approved.", verbose=verbose)
                    approved = "HYPOTHESIS NOT VERIFIED"  # Default to not verified when max attempts reached
                    break

                ui.print_info("🧪 Starting Scientist round...", verbose=verbose)
                # Add variable descriptions before every scientist round
                ui.print_info("📚 Adding variable descriptions to conversation context...", verbose=verbose)
                self.add_variable_descriptions_to_conversation(json_dictionary_path)
                scientist_success = self.scientist_round(verbose=verbose)
                
                # If Scientist's code execution failed, run Critic with debugging query
                if not scientist_success:
                    ui.print_warning("⚠️ Scientist code execution failed. Running Critic with debugging query...", verbose=verbose)
                    
                    # Enable debugging mode for Critic
                    self.critic.set_debugging_mode(True)
                    
                    # Run Critic in debugging mode
                    critic_status = self.critic_round(verbose=verbose)
                    
                    # Disable debugging mode for next iteration
                    self.critic.set_debugging_mode(False)
                else:
                    ui.print_info("🔍 Starting Critic round...", verbose=verbose)
                    critic_status = self.critic_round(verbose=verbose)

                self.conversation_manager.increment_turns()
                ui.print_info(f"✅ Turn {self.conversation_manager.get_turns()} completed. Status: {critic_status}", verbose=verbose)
                
                # Check if we should break the loop based on the status
                if critic_status == "HYPOTHESIS VERIFIED":
                    approved = critic_status
                    break
                elif critic_status == "NOT APPLICABLE":
                    # For NOT APPLICABLE, we should also break as there's no point continuing
                    approved = critic_status
                    break
                elif critic_status is None or critic_status == "":
                    # For debugging mode (no judgment provided), continue the loop
                    ui.print_info("🔄 Debugging guidance provided. Continuing conversation...", verbose=verbose)
                    continue
                # For HYPOTHESIS NOT VERIFIED, continue the loop (keep approved = None)
                
            except Exception as e:
                ui.print_error(f"Conversation loop error: {str(e)}", "conversation loop", verbose=verbose)
                ui.print_info("🔄 Attempting to continue with next iteration...", verbose=verbose)
                # Don't break - try to continue the conversation
                continue
        
        # If no decision was made, default to not verified
        if approved is None:
            approved = "HYPOTHESIS NOT VERIFIED"
        
        # If approved, capture the number of turns
        if approved == "HYPOTHESIS VERIFIED":
            num_turns_to_approve = self.conversation_manager.get_turns()

        return approved, num_turns_to_approve, num_turns_to_autoapprove
    
    def execute_agent_round(self, agent, agent_name: str, conversation_manager, 
                           special_code_conditions: Optional[Dict[str, str]] = None, 
                           return_approval: bool = False, approval_keyword: Optional[str] = None, 
                           return_success: bool = False, verbose: bool = True) -> Optional[Union[bool, str]]:
        """
        Common method to execute an agent round with shared logic.
        
        Args:
            agent: The agent instance to generate response
            agent_name: Name of the agent for display purposes
            conversation_manager: Conversation manager to use
            special_code_conditions: Dict with special conditions for code execution
            return_approval: Whether to return approval status
            approval_keyword: Keyword to check for approval
            return_success: Whether to return success/failure status
            verbose: Whether to use verbose output
        
        Returns:
            If return_approval is True, returns boolean approval status
            If return_success is True, returns boolean success status
            Otherwise returns None
        """
        ui.print_loading(f"{agent_name} is working...")
        
        try:
            # Generate response with error handling
            start_time = time.time()
            ui.print_info(f"🔄 Generating {agent_name} response...")
            response = agent.generate(
                conversation_manager.get_conversation_history()
            )
            generation_time = time.time() - start_time
            ui.print_info(f"⏱️ Response generated in {generation_time:.2f} seconds")
            
            if not response or not response.strip():
                raise ValueError(f"Empty response from {agent_name}")
            
            # Get current turn count for JSON logging
            current_turn = conversation_manager.get_turns() + 1
            
            # Execute code if present (except for critic agents)
            code_output = ""
            should_execute_code = "```python" in response and agent_name not in ["Critic", "Planning Agent"]
            
            # Apply special conditions if provided
            if special_code_conditions:
                for condition, action in special_code_conditions.items():
                    if condition in response:
                        if action == "skip":
                            should_execute_code = False
                            ui.print_info(f"📊 {action} - skipping code execution")
                            code_output = f"No code execution needed: {condition}"
                        elif action == "custom_message":
                            ui.print_info(f"📊 {condition} - skipping code execution")
                            code_output = f"No code execution needed: {condition}"
                        break
            
            if should_execute_code:
                ui.print_info(f"💻 Executing code from {agent_name} response...")
                try:
                    ui.print_info("🔍 About to call execute_code_from_response...")
                    execution_result = execute_code_from_response(self.code_executor, response)
                    ui.print_info("🔍 execute_code_from_response completed successfully")
                    
                    # Handle the raw result from code executor
                    executed_code = ""
                    if execution_result is None:
                        code_output = "No execution result"
                        ui.print_warning("⚠️ No execution result returned")
                        if return_success:
                            conversation_manager.add_agent_response(agent_name, response, code_output)
                            return False

                    elif isinstance(execution_result, str):
                        # Handle string error case
                        code_output = execution_result
                        ui.print_agent_response(agent_name, response, code_output)
                        ui.print_warning(f"⚠️ Code execution error")
                        
                        if return_success:
                            conversation_manager.add_agent_response(agent_name, response, code_output)
                            return False

                    elif isinstance(execution_result, dict):
                        # Handle dictionary result
                        executed_code = execution_result.get("code", "")
                        if execution_result.get("success", False):
                            code_output = execution_result.get("output", "")

                            if execution_result.get("error"):
                                code_output += f"\nWarnings: {execution_result['error']}"
                                ui.print_agent_response(agent_name, response, code_output)
                                ui.print_warning("⚠️ Code execution had warnings, but continuing...")

                            if return_success:
                                conversation_manager.add_agent_response(agent_name, response, code_output)

                        else:
                            code_output = f"Error: {execution_result.get('error', 'Unknown error')}"
                            ui.print_warning(f"⚠️ Code execution returned an error output")
                            ui.print_warning("⚠️ But continuing...")
                            ui.print_agent_response(agent_name, response, code_output)

                            if return_success:
                                conversation_manager.add_agent_response(agent_name, response, code_output)
                                return False
                    else:
                        code_output = str(execution_result)
                    

                except Exception as e:
                    ui.print_warning(f"⚠️ Code execution failed: {str(e)}")
                    code_output = f"Code execution failed: {str(e)}"
                    # If we're tracking success and code execution failed, return False
                    if return_success:
                        conversation_manager.add_agent_response(agent_name, response, code_output)
                        return False
            
            # Truncate conversation after adding response
            conversation_manager.truncate_conversation()
            
            # Log and display response
            ui.print_agent_response(agent_name, response, code_output)
            
            # Log to JSON for ALL agent responses
            if should_execute_code and execution_result is not None:
                if isinstance(execution_result, str):
                    # String error case
                    ui.log_agent_turn(current_turn, agent_name, response, 
                                    code_execution_success=False, 
                                    code_output="", 
                                    code_error=execution_result, 
                                    code_traceback="")
                elif isinstance(execution_result, dict):
                    # Dictionary result case
                    ui.log_agent_turn(current_turn, agent_name, response, 
                                    code_execution_success=execution_result.get("success", False), 
                                    code_output=execution_result.get("output", ""), 
                                    code_error=execution_result.get("error", ""), 
                                    code_traceback="")
                else:
                    # Fallback for other execution result types
                    ui.log_agent_turn(current_turn, agent_name, response)
            else:
                # No code execution
                ui.log_agent_turn(current_turn, agent_name, response)
            
            # Return approval status if requested
            if return_approval and approval_keyword:
                # First try to parse XML judgment (for critic responses)
                xml_judgment = parse_critic_judgment(response)
                if xml_judgment:
                    return xml_judgment
                
                # Fallback to keyword matching for backward compatibility
                if approval_keyword in response:
                    return "HYPOTHESIS VERIFIED"
                elif "NOT APPLICABLE" in response:
                    return "NOT APPLICABLE"
                else:
                    return "HYPOTHESIS NOT VERIFIED"
            
            # Return success status if requested
            if return_success:
                return True
            
            return None
            
        except (ValueError, RuntimeError) as e:
            # Re-raise critical errors that should stop the conversation
            error_msg = f"{agent_name} round failed: {str(e)}"
            ui.print_error(error_msg, agent_name.lower().replace(" ", "_"))
            # Return False for success status if requested
            if return_success:
                conversation_manager.add_agent_response(agent_name, response, code_output)
                return False
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # Handle other errors gracefully - don't re-raise
            error_msg = f"{agent_name} round failed: {str(e)}"
            ui.print_error(error_msg, agent_name.lower().replace(" ", "_"))
            # Add a placeholder response to keep the conversation flowing
            placeholder_response = f"{agent_name} encountered an error: {str(e)}. Please review the data and try again."
            conversation_manager.add_agent_response(agent_name, placeholder_response, "")
            ui.print_agent_response(agent_name, placeholder_response, "")
            # Return False for success status if requested
            if return_success:
                conversation_manager.add_agent_response(agent_name, response, code_output)
                return False
            return None
    
    def scientist_round(self, verbose: bool = True) -> bool:
        """Handle Scientist response round."""
        return self.execute_agent_round(
            agent=self.scientist,
            agent_name="Scientist",
            conversation_manager=self.conversation_manager,
            return_success=True,
            verbose=verbose
        )
    
    def critic_round(self, verbose: bool = True) -> str:
        """Handle Critic review round."""
        return self.execute_agent_round(
            agent=self.critic,
            agent_name="Critic",
            conversation_manager=self.conversation_manager,
            return_approval=True,
            approval_keyword="HYPOTHESIS VERIFIED",
            verbose=verbose
        )
    
    def planning_agent_round(self, hypothesis: str = None, filtered_csv_statistics: Dict[str, Any] = None, verbose: bool = True) -> None:
        """Handle Enhanced Planning Agent round."""
        if self.enhanced_planning_agent is None:
            ui.print_warning("⚠️ Planning Agent is not available (disabled in configuration)", verbose=verbose)
            return
        
        # Add hypothesis and dataset information to conversation context if provided
        if hypothesis or filtered_csv_statistics:
            context_message = "ENHANCED PLANNING CONTEXT:\n"
            
            if hypothesis:
                context_message += f"HYPOTHESIS TO TEST: {hypothesis}\n"
            
            if filtered_csv_statistics and 'error' not in filtered_csv_statistics:
                context_message += (
                    f"\nFILTERED DATASET INFORMATION (contains only relevant variables):\n"
                    f"- Total rows in filtered dataset: {filtered_csv_statistics['total_rows']:,}\n"
                    f"- Total columns in filtered dataset: {filtered_csv_statistics['total_cols']}\n"
                    f"- Missing values: {filtered_csv_statistics['missing_data']:,} ({filtered_csv_statistics['missing_percentage']}%)\n"
                    f"- Numeric columns: {filtered_csv_statistics['numeric_cols']}\n"
                    f"- Categorical columns: {filtered_csv_statistics['categorical_cols']}\n"
                )
                if filtered_csv_statistics.get('all_columns'):
                    context_message += f"- Available variables: {', '.join(filtered_csv_statistics['all_columns'])}\n"
            
            # Add context to conversation
            self.conversation_manager.messages.append({
                "role": "System",
                "content": context_message
            })
        
        # Use the existing enhanced planning agent
        self.execute_agent_round(
            agent=self.enhanced_planning_agent,
            agent_name="Planning Agent",
            conversation_manager=self.conversation_manager,
            verbose=verbose
        )
    
    def add_variable_descriptions_to_conversation(self, json_dictionary_path: str = None) -> None:
        """Add variable descriptions to conversation context for scientist/critic agents."""
        try:
            # Use the add_variable_descriptions_to_conversation function from simplified_data_processing
            add_variable_descriptions_to_conversation(
                self.conversation_manager,
                data_cache=self.data_cache
            )
        except Exception as e:
            ui.print_warning(f"⚠️ Could not add variable descriptions to conversation: {str(e)}")
    
    def print_experiment_summary(self, num_runs: int, total_hypotheses: int, total_approved: int, verbose: bool) -> None:
        """Print the final experiment summary."""
        ui.print_separator(verbose=verbose)
        ui.print_info(f"🎯 EXPERIMENT SUMMARY", verbose=verbose)
        ui.print_separator(verbose=verbose)
        overall_approval_rate = (total_approved / total_hypotheses) if total_hypotheses > 0 else 0
        ui.print_info(f"📊 Total runs: {num_runs}", verbose=verbose)
        ui.print_info(f"📝 Total hypotheses generated: {total_hypotheses}", verbose=verbose)
        ui.print_info(f"✅ Total hypotheses approved: {total_approved}", verbose=verbose)
        ui.print_info(f"📈 Overall approval rate: {overall_approval_rate:.2%}", verbose=verbose)
        ui.print_info(f"✅ All sampling runs completed.", verbose=verbose)
