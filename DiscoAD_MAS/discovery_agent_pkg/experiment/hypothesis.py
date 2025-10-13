"""
Hypothesis management module for the Discovery Agent.

This module handles all hypothesis-related operations including:
- Loading and sampling ADRD keywords
- Generating hypotheses from keywords
- Managing hypothesis tracking and storage
"""

import os
import random
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from ..ui.interface import ui
from ..agents.models import ModelClient


class HypothesisManager:
    """Manages hypothesis generation and tracking for the Discovery Agent."""
    
    def __init__(self, model_client: ModelClient):
        """Initialize the hypothesis manager with a model client."""
        self.model_client = model_client
        self.hypotheses = []  # Track generated hypotheses
    
    def load_adrd_keywords(self, keywords_file: str) -> List[str]:
        """
        Load ADRD keywords from a text file.
        
        Args:
            keywords_file (str): Path to the keywords file
        
        Returns:
            List[str]: List of loaded keywords, empty list if file not found
        """
        if not os.path.exists(keywords_file):
            ui.print_warning(f"🚫 Keywords file not found: {keywords_file}")
            return []
        
        try:
            with open(keywords_file, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
            ui.print_info(f"📚 Loaded {len(keywords)} ADRD keywords from {keywords_file}")
            return keywords
        except Exception as e:
            ui.print_error(f"Error loading keywords file: {str(e)}", "keywords_loading")
            return []
    
    def sample_keywords(self, keywords: List[str], num_keywords: int) -> List[str]:
        """
        Sample a random subset of keywords.
        
        Args:
            keywords (List[str]): Available keywords
            num_keywords (int): Number of keywords to sample
        
        Returns:
            List[str]: Sampled keywords
        """
        if num_keywords >= len(keywords):
            return keywords.copy()
        
        return random.sample(keywords, num_keywords)
    
    def generate_hypotheses_from_keywords(self, sampled_keywords: List[str], 
                                        dataset_dictionary: Optional[Dict[str, str]] = None,
                                        csv_statistics: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate multiple hypotheses from sampled keywords using LLM.
        
        Args:
            sampled_keywords (List[str]): Keywords to generate hypotheses from
            dataset_dictionary (Optional[Dict[str, str]]): Dataset variable descriptions
            csv_statistics (Optional[Dict[str, Any]]): Dataset statistics
        
        Returns:
            List[str]: Generated hypotheses
        """
        try:
            # Create context for hypothesis generation
            context = f"Generate 3-5 specific, testable hypotheses related to these ADRD keywords: {', '.join(sampled_keywords)}"
            
            if dataset_dictionary:
                context += f"\n\nAvailable variables in the dataset:\n"
                for var, desc in list(dataset_dictionary.items())[:20]:  # Limit to first 20 variables
                    context += f"- {var}: {desc}\n"
                if len(dataset_dictionary) > 20:
                    context += f"... and {len(dataset_dictionary) - 20} more variables\n"
            
            if csv_statistics and 'error' not in csv_statistics:
                context += f"\nDataset contains {csv_statistics.get('total_rows', 'unknown')} rows and {csv_statistics.get('total_cols', 'unknown')} columns."
            
            context += "\n\nEach hypothesis should be specific, measurable, and testable with the available data. Format each hypothesis on a new line starting with 'HYPOTHESIS:'"
            
            # Generate hypotheses using the model
            response = self.model_client.generate_response(
                messages=[{"role": "user", "content": context}],
                max_tokens=500,
                temperature=0.7
            )
            
            # Parse hypotheses from response
            hypotheses = []
            for line in response.split('\n'):
                if line.strip().startswith('HYPOTHESIS:'):
                    hypothesis = line.replace('HYPOTHESIS:', '').strip()
                    if hypothesis:
                        hypotheses.append(hypothesis)
            
            # If no hypotheses found, create a fallback
            if not hypotheses:
                fallback_hypothesis = f"Investigate the relationship between {', '.join(sampled_keywords)} in the dataset"
                hypotheses = [fallback_hypothesis]
            
            # Store hypotheses
            self.hypotheses.extend(hypotheses)
            
            return hypotheses
            
        except Exception as e:
            ui.print_error(f"Error generating hypotheses: {str(e)}", "hypothesis_generation")
            # Return fallback hypothesis
            fallback_hypothesis = f"Investigate the relationship between {', '.join(sampled_keywords)} in the dataset"
            return [fallback_hypothesis]
    
    def set_direct_hypothesis(self, hypothesis: str) -> List[str]:
        """
        Set a direct hypothesis (for direct hypothesis mode).
        
        Args:
            hypothesis (str): The hypothesis to set
        
        Returns:
            List[str]: List containing the single hypothesis
        """
        self.hypotheses = [hypothesis]
        return [hypothesis]
    
    def create_single_hypothesis_from_keywords(self, sampled_keywords: List[str]) -> List[str]:
        """
        Create a single hypothesis directly from sampled keywords (for direct mode).
        
        Args:
            sampled_keywords (List[str]): Keywords to create hypothesis from
        
        Returns:
            List[str]: List containing the single hypothesis
        """
        hypothesis = f"Investigate the relationship between {', '.join(sampled_keywords)} in the dataset"
        self.hypotheses = [hypothesis]
        return [hypothesis]
    
    def get_hypotheses(self) -> List[str]:
        """
        Get the current list of hypotheses.
        
        Returns:
            List[str]: Current hypotheses
        """
        return self.hypotheses.copy()
    
    def clear_hypotheses(self) -> None:
        """Clear the current hypotheses list."""
        self.hypotheses = []
    
    def add_hypothesis(self, hypothesis: str) -> None:
        """
        Add a hypothesis to the current list.
        
        Args:
            hypothesis (str): Hypothesis to add
        """
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)
    
    def get_hypothesis_count(self) -> int:
        """
        Get the number of current hypotheses.
        
        Returns:
            int: Number of hypotheses
        """
        return len(self.hypotheses)
    
    def load_hypotheses_from_csv(self, csv_file_path: str, hypothesis_column: str = "hypothesis") -> List[Dict[str, Any]]:
        """
        Load hypotheses from a CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file
            hypothesis_column (str): Name of the column containing hypotheses
        
        Returns:
            List[Dict[str, Any]]: List of hypothesis dictionaries with metadata
        """
        if not os.path.exists(csv_file_path):
            ui.print_error(f"❌ CSV file not found: {csv_file_path}", "csv_loading")
            return []
        
        try:
            # Try to read with pandas first (more robust)
            try:
                df = pd.read_csv(csv_file_path)
                ui.print_info(f"📊 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                
                # Check if hypothesis column exists
                if hypothesis_column not in df.columns:
                    available_cols = list(df.columns)
                    ui.print_error(f"❌ Column '{hypothesis_column}' not found in CSV. Available columns: {available_cols}", "csv_loading")
                    return []
                
                # Convert to list of dictionaries
                hypotheses_data = []
                for idx, row in df.iterrows():
                    hypothesis_data = {
                        'index': idx + 1,  # 1-based indexing
                        'hypothesis': str(row[hypothesis_column]).strip(),
                        'row_data': row.to_dict()  # Store all row data for potential use
                    }
                    if hypothesis_data['hypothesis']:  # Only add non-empty hypotheses
                        hypotheses_data.append(hypothesis_data)
                
                ui.print_info(f"📚 Loaded {len(hypotheses_data)} hypotheses from CSV file")
                return hypotheses_data
                
            except ImportError:
                # Fallback to csv module if pandas not available
                ui.print_warning("⚠️ pandas not available, using csv module")
                hypotheses_data = []
                
                with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for idx, row in enumerate(reader, 1):
                        if hypothesis_column in row and row[hypothesis_column].strip():
                            hypothesis_data = {
                                'index': idx,
                                'hypothesis': row[hypothesis_column].strip(),
                                'row_data': row
                            }
                            hypotheses_data.append(hypothesis_data)
                
                ui.print_info(f"📚 Loaded {len(hypotheses_data)} hypotheses from CSV file")
                return hypotheses_data
                
        except Exception as e:
            ui.print_error(f"Error loading CSV file: {str(e)}", "csv_loading")
            return []
    
    def update_csv_status(self, csv_file_path: str, hypothesis_index: int, status: str = "done", 
                         additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a hypothesis in the CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file
            hypothesis_index (int): Index of the hypothesis to update
            status (str): Status to set (e.g., "done", "processing", "failed")
            additional_data (Optional[Dict[str, Any]]): Additional data to add
        
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Check if hypothesis_index is valid
            if hypothesis_index < 1 or hypothesis_index > len(df):
                ui.print_error(f"❌ Invalid hypothesis index: {hypothesis_index}. Valid range: 1-{len(df)}", "csv_update")
                return False
            
            # Update the status (0-based indexing for pandas)
            row_idx = hypothesis_index - 1
            
            # Add or update status column
            if 'status' not in df.columns:
                df['status'] = 'pending'
            
            df.at[row_idx, 'status'] = status
            
            # Add additional data if provided
            if additional_data:
                for key, value in additional_data.items():
                    if key not in df.columns:
                        df[key] = None
                    df.at[row_idx, key] = value
            
            # Add timestamp
            from datetime import datetime
            if 'updated_at' not in df.columns:
                df['updated_at'] = None
            df.at[row_idx, 'updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Write back to CSV
            df.to_csv(csv_file_path, index=False)
            ui.print_info(f"✅ Updated hypothesis {hypothesis_index} status to '{status}' in CSV")
            return True
            
        except Exception as e:
            ui.print_error(f"Error updating CSV file: {str(e)}", "csv_update")
            return False
