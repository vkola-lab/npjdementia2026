"""
Hypothesis management module for the Discovery Agent.

This module handles all hypothesis-related operations including:
- Hypothesis generation from keywords and dataset information
- Keyword loading and sampling
- Hypothesis tracking and validation
"""

import os
import random
import ast
import re
from typing import List, Dict, Any, Optional
from .ui import ui
from .models import ModelClient, HypothesisGenerator


class HypothesisManager:
    """Handles all hypothesis-related operations for the Discovery Agent."""
    
    def __init__(self, model_client: ModelClient):
        """Initialize the hypothesis manager with a model client."""
        self.model_client = model_client
        self.hypothesis_generator = HypothesisGenerator(model_client)
        self.hypotheses = []
    
    def load_adrd_keywords(self, keywords_file_path: Optional[str] = None) -> List[str]:
        """Load ADRD keywords from the keywords file."""
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
    
    def _get_default_keywords(self) -> List[str]:
        """Get default ADRD keywords if file loading fails."""
        return [
            "age", "education", "gender", "MMSE", "MoCA", "CDR", "memory", "attention",
            "executive function", "language", "visuospatial", "diabetes", "hypertension",
            "stroke", "depression", "anxiety", "heart disease", "alcohol", "smoking",
            "Alzheimer's disease", "dementia", "mild cognitive impairment", "APOE",
            "family history", "obesity", "physical activity", "social support"
        ]
    
    def sample_keywords(self, keywords: List[str], sample_size: int = 3) -> List[str]:
        """Sample keywords from the collection."""
        if sample_size > len(keywords):
            sample_size = len(keywords)
        return random.sample(keywords, sample_size)
    
    def generate_hypotheses_from_keywords(self, sampled_keywords: List[str], dataset_dictionary: Optional[Dict[str, Any]] = None, csv_statistics: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate hypotheses from sampled keywords using the hypothesis generator.
        
        Args:
            sampled_keywords (List[str]): List of sampled keywords
            dataset_dictionary (Dict[str, Any], optional): Dictionary of available variables
            csv_statistics (Dict[str, Any], optional): Statistics from the CSV file
        
        Returns:
            List[str]: List of generated hypotheses
        """
        hypothesis_input = f"{', '.join(sampled_keywords)}"
        ui.print_info(f"📝 Generated hypothesis input: {hypothesis_input}")
        
        # Generate hypotheses with dataset information
        self.hypotheses = self.hypothesis_generator.generate(
            user_query=hypothesis_input,
            dataset_dictionary=dataset_dictionary,
            user_keywords=sampled_keywords,
            csv_statistics=csv_statistics
        )
        
        return self.hypotheses
    
    def create_single_hypothesis_from_keywords(self, sampled_keywords: List[str]) -> List[str]:
        """
        Create a single hypothesis directly from sampled keywords.
        
        Args:
            sampled_keywords (List[str]): List of sampled keywords
        
        Returns:
            List[str]: List containing a single hypothesis
        """
        hypothesis_input = f"{', '.join(sampled_keywords)}"
        ui.print_info(f"📝 Using sampled keywords directly as hypothesis: {hypothesis_input}")
        self.hypotheses = [hypothesis_input]
        return self.hypotheses
    
    def set_direct_hypothesis(self, hypothesis: str) -> List[str]:
        """
        Set a direct hypothesis provided by the user.
        
        Args:
            hypothesis (str): The direct hypothesis to use
        
        Returns:
            List[str]: List containing the direct hypothesis
        """
        if not hypothesis or not hypothesis.strip():
            ui.print_error("❌ Empty hypothesis provided", "direct_hypothesis")
            return []
        
        ui.print_info(f"📝 Setting direct hypothesis: {hypothesis}")
        self.hypotheses = [hypothesis.strip()]
        return self.hypotheses
    
    def get_hypotheses(self) -> List[str]:
        """Get the current list of hypotheses."""
        return self.hypotheses
    
    def clear_hypotheses(self) -> None:
        """Clear the current hypotheses list."""
        self.hypotheses = []
    
    def validate_hypothesis(self, hypothesis: str) -> bool:
        """
        Validate a hypothesis for basic requirements.
        
        Args:
            hypothesis (str): The hypothesis to validate
        
        Returns:
            bool: True if hypothesis is valid, False otherwise
        """
        if not hypothesis or not hypothesis.strip():
            return False
        
        # Basic validation - hypothesis should be at least 10 characters
        if len(hypothesis.strip()) < 10:
            return False
        
        # Should contain some meaningful words
        words = hypothesis.strip().split()
        if len(words) < 3:
            return False
        
        return True
    
    def format_hypothesis_for_display(self, hypothesis: str, index: int) -> str:
        """
        Format a hypothesis for display purposes.
        
        Args:
            hypothesis (str): The hypothesis to format
            index (int): The index of the hypothesis
        
        Returns:
            str: Formatted hypothesis string
        """
        # Truncate if too long
        if len(hypothesis) > 100:
            truncated = hypothesis[:97] + "..."
            return f"{index}. {truncated}"
        else:
            return f"{index}. {hypothesis}"
    
    def get_hypothesis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current hypotheses.
        
        Returns:
            Dict[str, Any]: Summary information about hypotheses
        """
        return {
            'total_hypotheses': len(self.hypotheses),
            'hypotheses': self.hypotheses,
            'valid_hypotheses': [h for h in self.hypotheses if self.validate_hypothesis(h)],
            'average_length': sum(len(h) for h in self.hypotheses) / len(self.hypotheses) if self.hypotheses else 0
        }
