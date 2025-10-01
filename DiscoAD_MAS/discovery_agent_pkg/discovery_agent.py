"""
Main Discovery Agent class.

This module contains the core DiscoveryAgent class that orchestrates
the entire discovery process with proper error handling and state management.
"""

import signal
import sys
import time
import os
import json
import re
import random
import csv
from typing import Optional, Dict, Any
import logging
import emoji
from colorama import Fore, Style, init

from .config import config as default_config
from .models import ModelClient, create_scientist, create_critic, create_planning_agent, HypothesisGenerator
from .conversation import ConversationManager
from .code_executor import CodeExecutor
from .ui import ui

# Import cleanup functions from utils
from .utils import cleanup_gpu_processes, signal_handler, sigtstp_handler, execute_code_from_response, cleanup_resources

# Initialize colorama for cross-platform color support
init(autoreset=True)


class DiscoveryAgentSystem:
    """Main Discovery Agent class."""
    
    def __init__(self, config=None):
        """Initialize the Discovery Agent with all components."""
        # Use provided config or fall back to default
        self.config = config if config is not None else default_config
        
        # Delete and recreate working directory at the start
        self._cleanup_working_directory(recreate=True)
        
        # Initialize CSV tracking
        self._initialize_csv_tracking()
        
        try:
            ui.print_info(f"{emoji.emojize('⚙️')} Initializing Discovery Agent...")
            
            # Initialize all components directly
            self.model_client = ModelClient(self.config)
            self.scientist = create_scientist(self.model_client)
            self.critic = create_critic(self.model_client)
            self.planning_agent = create_planning_agent(self.model_client)
            self.hypothesis_generator = HypothesisGenerator(self.model_client)
            self.conversation_manager = ConversationManager(self.model_client, self.config)
            self.code_executor = CodeExecutor(timeout_seconds=self.config.code_timeout)
            self.hypotheses = []
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGTSTP, sigtstp_handler)
            
            ui.print_success(f"{emoji.emojize('🎉')} Discovery Agent initialized successfully!")
        
        except Exception as e:
            ui.print_error(str(e), "initialization")
            raise
    
    def _generate_filename_with_config(self, base_name: str, extension: str = "", include_timestamp: bool = True) -> str:
        """
        Generate a filename that includes important config variables.
        
        Args:
            base_name (str): Base name for the file
            extension (str): File extension (e.g., '.csv', '.json')
            include_timestamp (bool): Whether to include timestamp in filename
            
        Returns:
            str: Generated filename with config variables
        """
        import datetime
        
        # Get important config variables
        config_parts = []
        
        # Model name (shortened)
        model_short = self.config.model_name.split('/')[-1].replace('-', '_') if hasattr(self.config, 'model_name') else 'unknown'
        config_parts.append(f"model_{model_short}")
        
        # Hypothesis mode
        if hasattr(self.config, 'hypothesis_mode'):
            config_parts.append(f"mode_{self.config.hypothesis_mode}")
        
        # Number of keywords
        if hasattr(self.config, 'num_keywords'):
            config_parts.append(f"kw_{self.config.num_keywords}")
        
        # Number of sampling runs
        if hasattr(self.config, 'num_sampling_runs'):
            config_parts.append(f"runs_{self.config.num_sampling_runs}")
        
        # Outer loop iterations
        if hasattr(self.config, 'outer_loop_iterations'):
            config_parts.append(f"outer_{self.config.outer_loop_iterations}")
        
        # Auto approve after attempts
        if hasattr(self.config, 'auto_approve_after_attempts'):
            config_parts.append(f"auto_{self.config.auto_approve_after_attempts}")
        
        # Join config parts
        config_str = "_".join(config_parts)
        
        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{config_str}_{timestamp}{extension}"
        else:
            filename = f"{base_name}_{config_str}{extension}"
        
        return filename

    def _cleanup_working_directory(self, recreate=True):
        """Clean up the working directory and optionally recreate it."""
        import shutil
        from pathlib import Path
        
        working_dir = Path("working")
        
        if recreate:
            # Delete and recreate the entire directory (for initialization)
            if working_dir.exists():
                try:
                    shutil.rmtree(working_dir)
                    ui.print_info(f"{emoji.emojize('🗑️')} Deleted existing working directory")
                except Exception as e:
                    ui.print_warning(f"⚠️ Could not delete working directory: {str(e)}")
            
            # Recreate working directory
            try:
                working_dir.mkdir(exist_ok=True)
                ui.print_info(f"{emoji.emojize('📁')} Created fresh working directory")
            except Exception as e:
                ui.print_error(f"Could not create working directory: {str(e)}", "working_directory_creation")
                raise
        else:
            # Just clean the contents (for between hypotheses)
            if working_dir.exists():
                try:
                    # Remove all files and subdirectories in the working directory
                    for item in working_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    ui.print_info(f"✅ Cleaned up working directory")
                except Exception as e:
                    ui.print_warning(f"⚠️ Could not fully clean working directory: {str(e)}")
            else:
                ui.print_info(f"📁 Working directory does not exist, no cleanup needed")
    
    def _initialize_csv_tracking(self):
        """Initialize CSV file for tracking individual hypothesis approval status and details."""
        from pathlib import Path
        import datetime
        
        # Use relative results directory (created in config.py)
        results_dir = Path("results")
        
        # Generate filename with config variables
        csv_filename = self._generate_filename_with_config("hypothesis_approval_tracking", ".csv")
        self.csv_file_path = results_dir / csv_filename
        
        # Create new CSV file with headers
        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['run_number', 'keywords', 'hypothesis_index', 'hypothesis_text', 'hypothesis_approved', 'outer_iterations_used', 'inner_iterations_used', 'turns_to_approve', 'max_outer_iterations', 'max_inner_iterations', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        ui.print_info(f"{emoji.emojize('📊')} Created new CSV tracking file: {self.csv_file_path}")
    
    def _write_csv_entry(self, run_number, keywords, hypothesis_index, hypothesis_text, outer_iterations_used, inner_iterations_used, hypothesis_approved, turns_to_approve=None):
        """Write an entry to the CSV tracking file for individual hypothesis."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        keywords_str = ', '.join(keywords)
        
        with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['run_number', 'keywords', 'hypothesis_index', 'hypothesis_text', 'hypothesis_approved', 'outer_iterations_used', 'inner_iterations_used', 'turns_to_approve', 'max_outer_iterations', 'max_inner_iterations', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'run_number': run_number,
                'keywords': keywords_str,
                'hypothesis_index': hypothesis_index,
                'hypothesis_text': hypothesis_text,
                'hypothesis_approved': hypothesis_approved,
                'outer_iterations_used': outer_iterations_used,
                'inner_iterations_used': inner_iterations_used,
                'turns_to_approve': turns_to_approve if hypothesis_approved else None,
                'max_outer_iterations': self.config.outer_loop_iterations,
                'max_inner_iterations': self.config.auto_approve_after_attempts,
                'timestamp': timestamp
            })
        
        # Create informative log message
        if hypothesis_approved and turns_to_approve is not None:
            log_msg = f"Approved after {outer_iterations_used} outer iterations, {inner_iterations_used} turns in final iteration"
        elif not hypothesis_approved:
            log_msg = f"Not approved after {outer_iterations_used} outer iterations, {inner_iterations_used} turns in final iteration"
        else:
            log_msg = "Not Approved"
        
        ui.print_info(f"{emoji.emojize('💾')} Recorded hypothesis {hypothesis_index} data to CSV: {log_msg}")
            
    def _initialize_sampling_run(self):
        """Initialize the sampling run by loading keywords and setting up tracking."""
        # Load ADRD keywords
        keywords = self._load_adrd_keywords(self.config.keywords_file)
        if not keywords:
            ui.print_warning(f"{emoji.emojize('🚫')} No ADRD keywords found. Exiting.")
            return None, None, None
        
        verbose = self.config.verbose
        ui.print_info(f"{emoji.emojize('📚')} Loaded {len(keywords)} ADRD keywords", verbose=verbose)
        
        # Get number of sampling runs from config or use default
        num_runs = getattr(self.config, 'num_sampling_runs', 10)
        ui.print_info(f"{emoji.emojize('🎯')} Running {num_runs} keyword sampling iterations", verbose=verbose)
        
        # Track results
        total_hypotheses = 0
        total_approved = 0
        
        return keywords, num_runs, verbose, total_hypotheses, total_approved

    def _run_sampling_iterations(self, keywords, num_runs, verbose, total_hypotheses, total_approved):
        """Run the main sampling iterations loop."""
        for run_idx in range(1, num_runs + 1):
            ui.print_separator(verbose=verbose)
            ui.print_info(f"{emoji.emojize('🎲')} Sampling Run {run_idx}/{num_runs}", verbose=verbose)
            
            # Sample keywords
            sampled_keywords = self._sample_keywords(keywords, self.config.num_keywords)
            ui.print_info(f"🎯 Sampled keywords: {', '.join(sampled_keywords)}", verbose=verbose)
            
            # Create hypothesis from sampled keywords
            hypothesis_input = f"{', '.join(sampled_keywords)}"
            ui.print_info(f"📝 Generated hypothesis: {hypothesis_input}", verbose=verbose)
            
            # Get hypothesis mode from config
            generate_hypotheses = (self.config.hypothesis_mode == "generate")
            
            if generate_hypotheses:
                # Generate multiple hypotheses from sampled keywords
                ui.print_info(f"{emoji.emojize('🔬')} Generating multiple hypotheses from sampled keywords...", verbose=verbose)
                self.hypotheses = self.hypothesis_generator.generate(hypothesis_input)
                ui.print_hypothesis_list(self.hypotheses, generated=True, verbose=verbose)
            else:
                # Use sampled keywords directly as a single hypothesis
                ui.print_info(f"{emoji.emojize('📝')} Using sampled keywords directly as hypothesis...", verbose=verbose)
                self.hypotheses = [hypothesis_input]
                ui.print_hypothesis_list(self.hypotheses, generated=False, verbose=verbose)
            
            # Process each generated hypothesis
            run_approved_count = 0
            for idx, hypothesis in enumerate(self.hypotheses, 1):
                approved = self._process_hypothesis(hypothesis, idx, run_idx, sampled_keywords, verbose)
                if approved:
                    run_approved_count += 1
            
            # Print run summary
            run_approval_rate = (run_approved_count / len(self.hypotheses)) if self.hypotheses else 0
            ui.print_info(f"{emoji.emojize('📊')} Run {run_idx} Summary:", verbose=verbose)
            ui.print_info(f"   🎯 Keywords: {', '.join(sampled_keywords)}", verbose=verbose)
            ui.print_info(f"   📝 Generated hypotheses: {len(self.hypotheses)}", verbose=verbose)
            ui.print_info(f"   ✅ Approved hypotheses: {run_approved_count}", verbose=verbose)
            ui.print_info(f"   📈 Approval rate: {run_approval_rate:.2%}", verbose=verbose)
            
            # Update totals
            total_hypotheses += len(self.hypotheses)
            total_approved += run_approved_count
        
        return total_hypotheses, total_approved

    def _process_hypothesis(self, hypothesis, idx, run_idx, sampled_keywords, verbose):
        """Process a single hypothesis through the outer loop: data prep -> planner -> scientist/critic -> if not approved, go back to data prep with different variables."""
        ui.print_progress(idx, len(self.hypotheses), verbose=verbose)
        ui.print_separator(verbose=verbose)
        ui.print_info(f"🔍 Working on hypothesis {idx}: {hypothesis}", verbose=verbose)
        
        # Track outer loop iterations
        outer_loop_iterations = self.config.outer_loop_iterations
        approved = False
        num_turns_to_approve = None
        num_turns_to_autoapprove = self.config.auto_approve_after_attempts
        previous_conversation_summary = None
        total_outer_iterations = 0
        final_iteration_turns = None
        
        for outer_iteration in range(1, outer_loop_iterations + 1):
            ui.print_separator(verbose=verbose)
            ui.print_info(f"{emoji.emojize('🔄')} OUTER LOOP ITERATION {outer_iteration}/{outer_loop_iterations}", verbose=verbose)
            
            # Reset conversation for each outer loop iteration
            self.conversation_manager.reset_conversation()
            
            # Add previous conversation summary as system message if available (at index 0)
            if previous_conversation_summary and outer_iteration > 1:
                ui.print_info(f"{emoji.emojize('📝')} Adding previous conversation summary as context...", verbose=verbose)
                self.conversation_manager.messages.append({
                    "role": "System", 
                    "content": f"PREVIOUS ITERATION SUMMARY:\n{previous_conversation_summary}\n\nThis is iteration {outer_iteration} of {outer_loop_iterations}. Use the above summary to avoid repeating the same approaches and try different variables or methods."
                })
            
            # Add user input (will be at index 0 if no summary, or index 1 if summary exists)
            self.conversation_manager.add_user_input(hypothesis)
            
            # Run manual preprocessing for this specific hypothesis and iteration
            ui.print_info(f"{emoji.emojize('🔧')} Running manual preprocessing for hypothesis {idx}, iteration {outer_iteration}...", verbose=verbose)
            try:
                preprocessing_success = self.run_manual_preprocessing_conversation(hypothesis, self.config.dataset_folder, previous_conversation_summary)
                if not preprocessing_success:
                    ui.print_warning(f"{emoji.emojize('⚠️')} Manual preprocessing failed for hypothesis {idx}, iteration {outer_iteration}, but continuing with analysis...", verbose=verbose)
            except Exception as e:
                ui.print_error(f"Manual preprocessing failed for hypothesis {idx}, iteration {outer_iteration}: {str(e)}", "manual_preprocessing", verbose=verbose)
                # Continue with the loop even if preprocessing fails
            
            # Run conversation until approved
            iteration_approved, iteration_turns_to_approve, iteration_turns_to_autoapprove = self._run_conversation_until_approved(verbose)
            
            # Track this iteration
            total_outer_iterations = outer_iteration
            final_iteration_turns = iteration_turns_to_approve if iteration_approved else iteration_turns_to_autoapprove
            
            ui.print_info(f"{emoji.emojize('📊')} Iteration {outer_iteration} result: {'✅ APPROVED' if iteration_approved else '❌ NOT APPROVED'}", verbose=verbose)
            
            if iteration_approved:
                # If approved, we're done with this hypothesis
                approved = True
                num_turns_to_approve = iteration_turns_to_approve
                ui.print_success(f"{emoji.emojize('🎉')} Hypothesis {idx} APPROVED in iteration {outer_iteration}!", verbose=verbose)
                break
            else:
                # If not approved, create summary for next iteration
                if outer_iteration < outer_loop_iterations:
                    ui.print_info(f"{emoji.emojize('📝')} Creating conversation summary for next iteration...", verbose=verbose)
                    previous_conversation_summary = self.conversation_manager.create_outer_loop_summary()
                    ui.print_info(f"{emoji.emojize('🔄')} Will try different approach in iteration {outer_iteration + 1}...", verbose=verbose)
                else:
                    ui.print_warning(f"{emoji.emojize('⚠️')} Hypothesis {idx} NOT APPROVED after {outer_loop_iterations} iterations", verbose=verbose)
        
        ui.print_approval_status(approved, verbose=verbose)
        
        # Write CSV entry for this hypothesis (only once, after all outer loop iterations are complete)
        self._write_csv_entry(
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
        ui.print_info(f"{emoji.emojize('🧹')} Cleaning up working directory after hypothesis {idx}...", verbose=verbose)
        self._cleanup_working_directory(recreate=False)
        
        return approved

    def _run_conversation_until_approved(self, verbose):
        """Run the conversation loop until a hypothesis is approved or max attempts reached."""
        approved = False
        num_turns_to_approve = None
        num_turns_to_autoapprove = self.config.auto_approve_after_attempts
                
        # Planning agent round (only on first turn)
        if self.conversation_manager.get_turns() == 0:
            ui.print_info("📋 Starting Planning Agent round...", verbose=verbose)
            self._planning_agent_round(verbose=verbose)

        while not approved:
            try:
                ui.print_info(f"🔄 Conversation turn {self.conversation_manager.get_turns() + 1}", verbose=verbose)
                
                if self.conversation_manager.should_auto_approve():
                    ui.print_warning(f"{emoji.emojize('⚠️')} Maximum attempts reached. Results not approved.", verbose=verbose)
                    approved = False # anyways by default
                    break

                ui.print_info("🧪 Starting Scientist round...", verbose=verbose)
                scientist_success = self._scientist_round(verbose=verbose)
                
                ui.print_info("🔍 Starting Critic round...", verbose=verbose)
                approved = self._critic_round(verbose=verbose)

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

    def _print_experiment_summary(self, num_runs, total_hypotheses, total_approved, verbose):
        """Print the final experiment summary."""
        ui.print_separator(verbose=verbose)
        ui.print_info(f"{emoji.emojize('🎯')} EXPERIMENT SUMMARY", verbose=verbose)
        ui.print_separator(verbose=verbose)
        overall_approval_rate = (total_approved / total_hypotheses) if total_hypotheses > 0 else 0
        ui.print_info(f"📊 Total runs: {num_runs}", verbose=verbose)
        ui.print_info(f"📝 Total hypotheses generated: {total_hypotheses}", verbose=verbose)
        ui.print_info(f"✅ Total hypotheses approved: {total_approved}", verbose=verbose)
        ui.print_info(f"📈 Overall approval rate: {overall_approval_rate:.2%}", verbose=verbose)
        ui.print_info(f"{emoji.emojize('✅')} All sampling runs completed.", verbose=verbose)

    def run(self):
        """Run the main discovery agent loop with keyword sampling."""
        try:
            # Initialize sampling run
            init_result = self._initialize_sampling_run()
            if init_result[0] is None:  # keywords is None
                return
            
            keywords, num_runs, verbose, total_hypotheses, total_approved = init_result
            
            # Run the main sampling iterations
            total_hypotheses, total_approved = self._run_sampling_iterations(
                keywords, num_runs, verbose, total_hypotheses, total_approved
            )
            
            # Print final experiment summary
            self._print_experiment_summary(num_runs, total_hypotheses, total_approved, verbose)
        except KeyboardInterrupt:
            ui.print_info(f"{emoji.emojize('🛑')} Interrupted by user. Cleaning up...")
            cleanup_resources()
        except Exception as e:
            ui.print_error(str(e), "main loop")
            cleanup_resources()
            raise

    def _execute_agent_round(self, agent, agent_name, conversation_manager, 
                           special_code_conditions=None, 
                           return_approval=False, approval_keyword=None, return_success=False, verbose=True):
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
                            ui.print_warning(f"⚠️ Code execution returned an error output: {code_output}")
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
            
    def _scientist_round(self, verbose=True):
        """Handle Scientist response round."""
        return self._execute_agent_round(
            agent=self.scientist,
            agent_name="Scientist",
            conversation_manager=self.conversation_manager,
            return_success=True,
            verbose=verbose
        )
    
    
    def _critic_round(self, verbose=True):
        """Handle Critic review round."""
        return self._execute_agent_round(
            agent=self.critic,
            agent_name="Critic",
            conversation_manager=self.conversation_manager,
            return_approval=True,
            approval_keyword="RESULTS APPROVED",
            verbose=verbose
        )
    
    def _planning_agent_round(self, verbose=True):
        """Handle Planning Agent round."""
        self._execute_agent_round(
            agent=self.planning_agent,
            agent_name="Planning Agent",
            conversation_manager=self.conversation_manager,
            verbose=verbose
        )
    
    def get_relevant_variables(self, hypothesis: str, json_path: str, previous_iteration_context: str = None) -> Dict[str, str]:
        """
        Get all relevant variables for a hypothesis from a JSON dictionary.
        
        This method uses an LLM to analyze a hypothesis and identify all relevant
        variables from a data dictionary. It returns a simple dictionary with
        variable names as keys and their descriptions as values.
        
        Args:
            hypothesis (str): The hypothesis to analyze
            json_path (str): Path to the JSON dictionary file
            previous_iteration_context (str, optional): Context from previous outer loop iteration
        
        Returns:
            Dict[str, str]: Dictionary with variable names as keys and descriptions as values
            
        Example:
            >>> agent = DiscoveryAgentSystem()
            >>> variables = agent.get_relevant_variables(
            ...     "Cognitive decline is associated with age and education",
            ...     "/path/to/dictionary.json"
            ... )
            >>> print(variables)
            {'AGE': 'Age at visit', 'EDUC': 'Years of education', ...}
        """
        
        # Load the JSON dictionary
        with open(json_path, 'r') as f:
            dictionary = json.load(f)
        
        # Use the instance's model client
        model_client = self.model_client
        
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
                f"IMPORTANT: Based on the previous iteration context above, try to select slightly DIFFERENT variables this time. "
                f"If the previous iteration used certain variables that didn't work well or led to issues, "
                f"consider alternative variables that might provide better insights for testing this hypothesis. "
                f"Focus on variables that haven't been extensively used before or that might offer a different perspective.\n\n"
            )
        
        prompt = (
            f"You are a data analysis expert. Given a hypothesis and a dictionary of data fields, identify the *smallest possible set* of keys that are strictly necessary for testing or analyzing this hypothesis. "
            "Choose as few keys as possible to keep the experiment clear and focused. Avoid including extra or tangentially related variables."
            f"{context_section}"
            f'Hypothesis: "{hypothesis}"\n\n'
            f"Available data fields:\n"
            f"{keys_text}\n\n"
            "Please identify only the most essential keys for this hypothesis. For each key, provide:\n"
            "1. The key name\n"
            "2. A brief explanation of why it's strictly necessary\n\n"
            "Format your response as a numbered list with the key name and explanation.\n\n"
            "At the end of your response, provide a Python list of only the variable names you identified, formatted as:\n"
            'variable_names = ["key1", "key2", "key3", ...]'
        )

        # Generate response using the model client
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = model_client.generate(messages)
        print(f"LLM Response: {response}")
        
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
    
    def filter_csv_with_dictionary(self, folder_path: str, relevant_variables: Dict[str, str], hypothesis: str) -> tuple[str, Dict[str, str]]:
        """
        Filter CSV files in a folder using relevant variables from dictionary and save to .working.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            relevant_variables (Dict[str, str]): Dictionary of relevant variable names and descriptions
            hypothesis (str): The hypothesis being tested
        
        Returns:
            tuple[str, Dict[str, str]]: Path to the filtered CSV file and dictionary of variable datatypes
        """
        import pandas as pd
        import os
        from pathlib import Path
        
        ui.print_info(f"{emoji.emojize('🔍')} Filtering CSV files with relevant variables...")
        
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
            
            # Create .working directory if it doesn't exist
            working_dir = Path("working")
            working_dir.mkdir(exist_ok=True)
            
            # # Create a safe filename based on hypothesis
            # safe_hypothesis = "".join(c for c in hypothesis[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            # safe_hypothesis = safe_hypothesis.replace(' ', '_')
            
            # Save filtered CSV to .working directory
            output_filename = f"filtered_dataset.csv"
            output_path = working_dir / output_filename
            
            filtered_df.to_csv(output_path, index=False)
            
            ui.print_success(f"✅ Filtered CSV saved to: {output_path}")
            ui.print_info(f"📊 Filtered CSV has {len(filtered_df)} rows and {len(filtered_df.columns)} columns")
            
            # Get datatypes of the relevant variables using column names as keys
            variable_datatypes = {}
            for col in filtered_df.columns:
                dtype = str(filtered_df[col].dtype)
                variable_datatypes[col] = dtype
            
            return str(output_path), variable_datatypes
            
        except Exception as e:
            error_msg = f"Error filtering CSV file: {str(e)}"
            ui.print_error(error_msg, "csv_filtering")
            raise RuntimeError(error_msg) from e
    
    def run_manual_preprocessing_conversation(self, hypothesis: str, dataset_folder: str = None, previous_iteration_context: str = None):
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
        """
        ui.print_info(f"{emoji.emojize('🔧')} Starting manual preprocessing conversation...")
        
        try:
            # Use provided dataset folder or get from user
            if dataset_folder:
                folder_path = dataset_folder
                ui.print_info(f"📁 Using dataset folder from config: {folder_path}")
            else:
                folder_path = ui.get_folder_path("Enter the path to the folder containing your dataset (CSV and JSON files)")
            
            # Auto-detect CSV and JSON files in the folder
            csv_path, json_path = self._detect_dataset_files(folder_path)
            
            if not hypothesis.strip():
                ui.print_warning(f"{emoji.emojize('⚠️')} No hypothesis provided. Exiting manual preprocessing.")
                return False
            
            ui.print_info(f"🔬 Hypothesis: {hypothesis}")
            ui.print_info(f"📁 Dataset folder: {folder_path}")
            ui.print_info(f"📊 CSV file: {csv_path}")
            ui.print_info(f"📚 JSON dictionary: {json_path}")
            
            # Get relevant variables from dictionary
            ui.print_info(f"{emoji.emojize('🔍')} Analyzing hypothesis to find relevant variables...")
            if previous_iteration_context:
                ui.print_info(f"{emoji.emojize('🔄')} Using previous iteration context to guide variable selection...")
            relevant_variables = self.get_relevant_variables(hypothesis, json_path, previous_iteration_context)
            
            if not relevant_variables:
                ui.print_warning(f"{emoji.emojize('⚠️')} No relevant variables found. Using all available data.")
                # Create a dummy dictionary with all columns (we'll handle this in filtering)
                relevant_variables = {"ALL_COLUMNS": "All available columns"}
            
            ui.print_info(f"🎯 Found {len(relevant_variables)} relevant variables:")
            for var_name, description in relevant_variables.items():
                ui.print_info(f"   • {var_name}: {description}")

            # Filter CSV files using relevant variables
            filtered_csv_path, variable_datatypes = self.filter_csv_with_dictionary(folder_path, relevant_variables, hypothesis)
            
            variable_datatypes_str = "\n".join([f"- {col}: {dtype}" for col, dtype in variable_datatypes.items()])

            summarized_messages = [{
                "role": "System",
                "content": (
                    f"Filtered data saved to: {filtered_csv_path}\n\n"
                    f"Relevant variables for this hypothesis:\n{variable_datatypes_str}"
                )
            }]
            self.conversation_manager.messages.extend(summarized_messages)

            ui.print_info(f"{emoji.emojize('✅')} Manual preprocessing completed successfully!")
            ui.print_info(f"📊 Filtered data saved to: {filtered_csv_path}")
            return True
            
        except Exception as e:
            error_msg = f"Manual preprocessing failed: {str(e)}"
            ui.print_error(error_msg, "manual_preprocessing")
            return False

    def _detect_dataset_files(self, folder_path: str) -> tuple:
        """
        Auto-detect CSV and JSON files in the given folder.
        
        Args:
            folder_path (str): Path to folder containing dataset files
        
        Returns:
            tuple: (csv_path, json_path) - paths to the detected files
        
        Raises:
            ValueError: If required files are not found
        """
        import os
        
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
            import json
            with open(json_path, 'r') as f:
                json.load(f)
            ui.print_success(f"✅ JSON dictionary file is valid: {json_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_file}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file {json_file}: {str(e)}")
        
        return csv_path, json_path
    
    def _load_adrd_keywords(self, keywords_file_path=None):
        """Load ADRD keywords from the keywords file."""
        import random
        import os
        
        if keywords_file_path is None:
            keywords_file_path = "adrd_keywords.txt"
        
        # Check if keywords file exists at the specified path
        if not os.path.exists(keywords_file_path):
            # Try to find it in the parent directory
            parent_keywords_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), keywords_file_path)
            if os.path.exists(parent_keywords_file):
                keywords_file_path = parent_keywords_file
            else:
                ui.print_warning(f"⚠️ Keywords file not found at {keywords_file_path}. Using default keywords.")
                return self._get_default_keywords()
        
        try:
            with open(keywords_file_path, 'r') as f:
                lines = f.readlines()
            
            # Filter out comments and empty lines
            keywords = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    keywords.append(line)
            
            ui.print_info(f"📚 Loaded {len(keywords)} keywords from {keywords_file_path}")
            return keywords
            
        except Exception as e:
            ui.print_error(f"Error loading keywords: {str(e)}", "keyword_loading")
            return self._get_default_keywords()
    
    def _get_default_keywords(self):
        """Get default ADRD keywords if file loading fails."""
        return [
            "age", "education", "gender", "MMSE", "MoCA", "CDR", "memory", "attention",
            "executive function", "language", "visuospatial", "diabetes", "hypertension",
            "stroke", "depression", "anxiety", "heart disease", "alcohol", "smoking",
            "Alzheimer's disease", "dementia", "mild cognitive impairment", "APOE",
            "family history", "obesity", "physical activity", "social support"
        ]
    
    def _sample_keywords(self, keywords, sample_size=3):
        """Sample keywords from the collection."""
        import random
        if sample_size > len(keywords):
            sample_size = len(keywords)
        return random.sample(keywords, sample_size)


def main_discovery_agent(config=None):
    """Main entry point for the Discovery Agent."""
    try:
        # Print banner
        ui.print_banner()
        
        # Initialize and run agent with better error handling
        ui.print_info("🔧 Initializing Discovery Agent System...")
        agent = DiscoveryAgentSystem(config)
        ui.print_info("🚀 Starting Discovery Agent...")
        agent.run()
        
    except KeyboardInterrupt:
        ui.print_info("🛑 Interrupted by user.")
        cleanup_resources()
    except ImportError as e:
        ui.print_error(f"Import error: {str(e)}", "main")
        ui.print_info("💡 Check if all required packages are installed.")
        cleanup_resources()
        sys.exit(1)
    except OSError as e:
        ui.print_error(f"System error: {str(e)}", "main")
        ui.print_info("💡 Check GPU availability and CUDA setup.")
        cleanup_resources()
        sys.exit(1)
    except RuntimeError as e:
        ui.print_error(f"Runtime error: {str(e)}", "main")
        ui.print_info("💡 This might be a model loading or GPU memory issue.")
        cleanup_resources()
        sys.exit(1)
    except Exception as e:
        ui.print_error(f"Unexpected error: {str(e)}", "main")
        ui.print_info("💡 Check the error details above for troubleshooting.")
        import traceback
        ui.print_error(f"Full traceback: {traceback.format_exc()}", "debug")
        cleanup_resources()
        sys.exit(1)
    finally:
        ui.print_info("👋 Discovery Agent session ended.")
        ui.close_log()
