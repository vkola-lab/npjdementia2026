"""
Experiment runner module for the Discovery Agent.

This module handles all experiment-related operations including:
- Sampling run orchestration
- Hypothesis processing and iteration management
- Experiment statistics and reporting
- Outer loop iteration management
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from .ui import ui
from .conversation import ConversationManager
from .code_executor import CodeExecutor
from .models import ModelClient, create_scientist, create_critic, create_enhanced_planning_agent
from .simplified_data_processing import SimplifiedDataProcessor
from .hypothesis_management import HypothesisManager
from .file_management import FileManager
from .utils import execute_code_from_response


class ExperimentRunner:
    """Handles all experiment execution operations for the Discovery Agent."""
    
    def __init__(self, model_client: ModelClient, config=None):
        """Initialize the experiment runner with all required components."""
        self.config = config
        self.model_client = model_client
        
        # Initialize all components
        self.data_processor = SimplifiedDataProcessor(model_client)
        self.hypothesis_manager = HypothesisManager(model_client)
        self.file_manager = FileManager(config)
        
        # Initialize agents
        self.scientist = create_scientist(model_client)
        self.critic = create_critic(model_client)
        self.enhanced_planning_agent = create_enhanced_planning_agent(model_client)
        
        # Initialize conversation and code execution
        self.conversation_manager = ConversationManager(model_client, config)
        self.code_executor = CodeExecutor(timeout_seconds=config.code_timeout if config else 30)
    
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
            ui.print_separator(verbose=verbose)
            
            # Handle direct hypothesis mode
            if keywords is None:  # Direct hypothesis mode
                ui.print_info(f"🎯 Direct Hypothesis Run {run_idx}/{num_runs}", verbose=verbose)
                sampled_keywords = []  # No keywords needed for direct hypothesis
            else:
                ui.print_info(f"🎲 Sampling Run {run_idx}/{num_runs}", verbose=verbose)
                # Sample keywords
                sampled_keywords = self.hypothesis_manager.sample_keywords(keywords, self.config.num_keywords)
                ui.print_info(f"🎯 Sampled keywords: {', '.join(sampled_keywords)}", verbose=verbose)
            # Create hypothesis from sampled keywords (for non-direct modes)
                hypothesis_input = f"{', '.join(sampled_keywords)}"
                ui.print_info(f"📝 Generated hypothesis: {hypothesis_input}", verbose=verbose) ######## can be moved into above else statement
            
            # if keywords is not None:
            #     hypothesis_input = f"{', '.join(sampled_keywords)}"
            #     ui.print_info(f"📝 Generated hypothesis: {hypothesis_input}", verbose=verbose) ######## can be moved into above else statement
            
            # Get hypothesis mode from config
            hypothesis_mode = self.config.hypothesis_mode
            
            if hypothesis_mode == "generate":
                # Generate multiple hypotheses from sampled keywords
                ui.print_info(f"🔬 Generating multiple hypotheses from sampled keywords...", verbose=verbose)
                
                # Load dataset dictionary and CSV statistics if available
                dataset_dictionary = None
                csv_statistics = None
                if hasattr(self.config, 'dataset_folder') and self.config.dataset_folder:
                    try:
                        # Auto-detect dataset files
                        csv_path, json_path = self.data_processor.detect_dataset_files(self.config.dataset_folder)
                        # Load the JSON dictionary
                        import json
                        with open(json_path, 'r') as f:
                            dataset_dictionary = json.load(f)
                        
                        # Load CSV to get real statistics using the reusable function
                        csv_statistics = self.data_processor.get_dataset_statistics(csv_path)
                        
                        ui.print_info(f"📚 Loaded dataset dictionary with {len(dataset_dictionary)} variables", verbose=verbose)
                        if 'error' not in csv_statistics:
                            ui.print_info(f"📊 CSV statistics: {csv_statistics['total_rows']:,} rows, {csv_statistics['total_cols']} cols, {csv_statistics['missing_percentage']}% missing", verbose=verbose)
                        else:
                            ui.print_warning(f"⚠️ Could not load CSV statistics: {csv_statistics['error']}", verbose=verbose)
                    except Exception as e:
                        ui.print_warning(f"⚠️ Could not load dataset dictionary or CSV: {str(e)}", verbose=verbose)
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
            else:  # "direct" mode
                # Use sampled keywords directly as a single hypothesis
                ui.print_info(f"📝 Using sampled keywords directly as hypothesis...", verbose=verbose)
                hypotheses = self.hypothesis_manager.create_single_hypothesis_from_keywords(sampled_keywords)
                ui.print_hypothesis_list(hypotheses, generated=False, verbose=verbose)
            
            # Process each generated hypothesis
            run_approved_count = 0
            for idx, hypothesis in enumerate(hypotheses, 1):
                approved = self.process_hypothesis(hypothesis, idx, run_idx, sampled_keywords, verbose)
                if approved:
                    run_approved_count += 1
            
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
        
        return total_hypotheses, total_approved
    
    def process_hypothesis(self, hypothesis: str, idx: int, run_idx: int, 
                          sampled_keywords: List[str], verbose: bool) -> bool:
        """
        Process a single hypothesis through the outer loop: data prep -> planner -> scientist/critic -> if not approved, go back to data prep with different variables.
        
        Args:
            hypothesis (str): The hypothesis to process
            idx (int): Index of the hypothesis
            run_idx (int): Run index
            sampled_keywords (List[str]): Keywords used for this hypothesis
            verbose (bool): Whether to use verbose output
        
        Returns:
            bool: True if hypothesis was approved, False otherwise
        """
        ui.print_progress(idx, len(self.hypothesis_manager.get_hypotheses()), verbose=verbose)
        ui.print_separator(verbose=verbose)
        ui.print_info(f"🔍 Working on hypothesis {idx}: {hypothesis}", verbose=verbose)
        
        # Track outer loop iterations
        outer_loop_iterations = self.config.outer_loop_iterations
        approved = False
        num_turns_to_approve = None
        num_turns_to_autoapprove = self.config.auto_approve_after_attempts
        previous_conversation_summary = None
        previously_used_variables = []  # Track variables used in previous iterations
        total_outer_iterations = 0
        final_iteration_turns = None
        
        for outer_iteration in range(1, outer_loop_iterations + 1):
            ui.print_separator(verbose=verbose)
            ui.print_info(f"🔄 OUTER LOOP ITERATION {outer_iteration}/{outer_loop_iterations}", verbose=verbose)
            
            # Reset conversation for each outer loop iteration
            self.conversation_manager.reset_conversation()
            
            # Add previous conversation summary as system message if available (at index 0)
            if previous_conversation_summary and outer_iteration > 1:
                ui.print_info(f"📝 Adding previous conversation summary as context...", verbose=verbose)
                self.conversation_manager.messages.append({
                    "role": "System", 
                    "content": f"PREVIOUS ITERATION SUMMARY:\n{previous_conversation_summary}\n\nThis is iteration {outer_iteration} of {outer_loop_iterations}. Use the above summary to avoid repeating the same approaches and try different variables or methods."
                })
            
            # Add hypothesis as user input (will be at index 0 if no summary, or index 1 if summary exists)
            self.conversation_manager.add_user_input(hypothesis)
            
            # Run manual preprocessing for this specific hypothesis and iteration
            ui.print_info(f"🔧 Running manual preprocessing for hypothesis {idx}, iteration {outer_iteration}...", verbose=verbose)
            if previously_used_variables:
                ui.print_info(f"📋 Passing {len(previously_used_variables)} previously used variables to avoid repetition", verbose=verbose)
            
            try:
                # Check if dataset_folder is available, otherwise pass None
                dataset_folder = getattr(self.config, 'dataset_folder', None) if hasattr(self.config, 'dataset_folder') else None
                preprocessing_result = self.data_processor.run_simplified_preprocessing(
                    hypothesis, dataset_folder, previous_conversation_summary, self.conversation_manager, previously_used_variables
                )
                if not preprocessing_result.get('success', False):
                    ui.print_warning(f"⚠️ Manual preprocessing failed for hypothesis {idx}, iteration {outer_iteration}, but continuing with analysis...", verbose=verbose)
                    # Set default values for failed preprocessing
                    filtered_csv_statistics = None
                    relevant_variables = None
                else:
                    # Extract information for enhanced planning
                    filtered_csv_statistics = preprocessing_result.get('filtered_csv_statistics')
                    relevant_variables = preprocessing_result.get('relevant_variables')
                    
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
                # Continue with the loop even if preprocessing fails
            
            # Run conversation until approved
            iteration_approved, iteration_turns_to_approve, iteration_turns_to_autoapprove = self.run_conversation_until_approved(
                verbose, 
                hypothesis, 
                filtered_csv_statistics, 
                relevant_variables
            )
            
            # Track this iteration
            total_outer_iterations = outer_iteration
            final_iteration_turns = iteration_turns_to_approve if iteration_approved else iteration_turns_to_autoapprove
            
            ui.print_info(f"📊 Iteration {outer_iteration} result: {'✅ APPROVED' if iteration_approved else '❌ NOT APPROVED'}", verbose=verbose)
            
            if iteration_approved:
                # If approved, we're done with this hypothesis
                approved = True
                num_turns_to_approve = iteration_turns_to_approve
                ui.print_success(f"🎉 Hypothesis {idx} APPROVED in iteration {outer_iteration}!", verbose=verbose)
                break
            else:
                # If not approved, create summary for next iteration
                if outer_iteration < outer_loop_iterations:
                    ui.print_info(f"📝 Creating conversation summary for next iteration...", verbose=verbose)
                    previous_conversation_summary = self.conversation_manager.create_outer_loop_summary()
                    ui.print_info(f"🔄 Will try different approach in iteration {outer_iteration + 1}...", verbose=verbose)
                else:
                    ui.print_warning(f"⚠️ Hypothesis {idx} NOT APPROVED after {outer_loop_iterations} iterations", verbose=verbose)
        
        ui.print_approval_status(approved, verbose=verbose)
        
        # Write CSV entry for this hypothesis (only once, after all outer loop iterations are complete)
        self.file_manager.write_csv_entry(
            run_number=run_idx,
            keywords=sampled_keywords,
            hypothesis_index=idx,
            hypothesis_text=hypothesis,
            outer_iterations_used=total_outer_iterations,  # This will be the final iteration number
            inner_iterations_used=final_iteration_turns,
            hypothesis_approved=approved,
            turns_to_approve=num_turns_to_approve
        )
        
        # Clean up working directory after each hypothesis (approved or not)
        ui.print_info(f"🧹 Cleaning up working directory after hypothesis {idx}...", verbose=verbose)
        self.file_manager.cleanup_working_directory(recreate=False)
        
        return approved
    
    def run_conversation_until_approved(self, verbose: bool, hypothesis: str = None, 
                                      filtered_csv_statistics: Dict[str, Any] = None, 
                                      relevant_variables: Dict[str, str] = None) -> Tuple[bool, Optional[int], int]:
        """
        Run the conversation loop until a hypothesis is approved or max attempts reached.
        
        Args:
            verbose (bool): Whether to use verbose output
            hypothesis (str): The hypothesis being tested
            filtered_csv_statistics (Dict[str, Any]): Statistics from filtered dataset
            relevant_variables (Dict[str, str]): Relevant variables for the hypothesis
        
        Returns:
            Tuple[bool, Optional[int], int]: (approved, turns_to_approve, turns_to_autoapprove)
        """
        approved = False
        num_turns_to_approve = None
        num_turns_to_autoapprove = self.config.auto_approve_after_attempts ### Do not need in this loop
                
        # Planning agent round (only on first turn)
        if self.conversation_manager.get_turns() == 0:
            ui.print_info("📋 Starting Planning Agent round...", verbose=verbose)
            self.planning_agent_round(hypothesis, filtered_csv_statistics, verbose=verbose)

        while not approved:
            try:
                ui.print_info(f"🔄 Conversation turn {self.conversation_manager.get_turns() + 1}", verbose=verbose)
                
                if self.conversation_manager.should_auto_approve():
                    ui.print_warning(f"⚠️ Maximum attempts reached. Results not approved.", verbose=verbose)
                    approved = False # anyways by default
                    break

                ui.print_info("🧪 Starting Scientist round...", verbose=verbose)
                scientist_success = self.scientist_round(verbose=verbose)
                
                ui.print_info("🔍 Starting Critic round...", verbose=verbose)
                approved = self.critic_round(verbose=verbose)

                self.conversation_manager.increment_turns()
                ui.print_info(f"✅ Turn {self.conversation_manager.get_turns()} completed. Approved: {approved}", verbose=verbose)
                
                
            except Exception as e:
                ui.print_error(f"Conversation loop error: {str(e)}", "conversation loop", verbose=verbose)
                ui.print_info("🔄 Attempting to continue with next iteration...", verbose=verbose)
                # Don't break - try to continue the conversation
                continue
        
        # If approved, capture the number of turns
        if approved:
            num_turns_to_approve = self.conversation_manager.get_turns()

        return approved, num_turns_to_approve, num_turns_to_autoapprove
    
    def execute_agent_round(self, agent, agent_name: str, conversation_manager, 
                           special_code_conditions: Optional[Dict[str, str]] = None, 
                           return_approval: bool = False, approval_keyword: Optional[str] = None, 
                           return_success: bool = False, verbose: bool = True) -> Optional[bool]:
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
                raise ValueError(f"Empty response from {agent_name}") #### Maybe redundant check because it is already in ModelClient.generate
            
            # Execute code if present (except for critic agents)
            code_output = ""
            should_execute_code = "```python" in response and agent_name not in ["Data Critic", "Critic", "Planning Agent"]
            
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
                        ui.print_warning(f"⚠️ Code execution error: {execution_result}")
                        
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
                            ui.print_warning(f"⚠️ Code execution returned an error output: {code_output}") ###################### Add if statement to only print warnings if error field is not empty
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
            
            # Return approval status if requested
            if return_approval and approval_keyword:
                return approval_keyword in response
            
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
    
    def critic_round(self, verbose: bool = True) -> bool:
        """Handle Critic review round."""
        return self.execute_agent_round(
            agent=self.critic,
            agent_name="Critic",
            conversation_manager=self.conversation_manager,
            return_approval=True,
            approval_keyword="RESULTS APPROVED",
            verbose=verbose
        )
    
    def planning_agent_round(self, hypothesis: str = None, filtered_csv_statistics: Dict[str, Any] = None, verbose: bool = True) -> None:
        """Handle Enhanced Planning Agent round."""
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
            agent_name="Enhanced Planning Agent",
            conversation_manager=self.conversation_manager,
            verbose=verbose
        )
    
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
