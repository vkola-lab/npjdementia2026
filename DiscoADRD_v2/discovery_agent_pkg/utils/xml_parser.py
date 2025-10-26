"""
XML parser utility for extracting structured data from agent responses.
"""

import re
from typing import Optional


def parse_critic_judgment(response: str) -> Optional[str]:
    """
    Parse the critic's judgment from XML tags in the response.
    
    Args:
        response (str): The critic's response text
        
    Returns:
        Optional[str]: The judgment value ("HYPOTHESIS VERIFIED", "HYPOTHESIS NOT VERIFIED", 
                      "NOT APPLICABLE") or None if not found
    """
    if not response or not isinstance(response, str):
        return None
    
    # Look for judgment XML tags
    judgment_pattern = r'<judgment>(.*?)</judgment>'
    matches = re.findall(judgment_pattern, response, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Get the last match (in case there are multiple)
        judgment = matches[-1].strip().upper()
        
        # Validate the judgment value
        valid_judgments = {
            "HYPOTHESIS VERIFIED",
            "HYPOTHESIS NOT VERIFIED", 
            "NOT APPLICABLE"
        }
        
        if judgment in valid_judgments:
            return judgment
    
    return None


def parse_xml_tag(response: str, tag_name: str) -> Optional[str]:
    """
    Parse any XML tag from a response.
    
    Args:
        response (str): The response text
        tag_name (str): The name of the XML tag to extract
        
    Returns:
        Optional[str]: The content within the XML tags or None if not found
    """
    if not response or not isinstance(response, str) or not tag_name:
        return None
    
    # Create pattern for the specific tag
    pattern = rf'<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>'
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Return the last match (in case there are multiple)
        return matches[-1].strip()
    
    return None


def has_xml_judgment(response: str) -> bool:
    """
    Check if the response contains a judgment XML tag.
    
    Args:
        response (str): The response text
        
    Returns:
        bool: True if judgment XML tag is found, False otherwise
    """
    return parse_critic_judgment(response) is not None
