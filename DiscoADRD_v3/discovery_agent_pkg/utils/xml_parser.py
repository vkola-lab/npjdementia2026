"""
XML parser utility for extracting structured data from agent responses.
"""

import re
from typing import Optional


def parse_critic_judgment(response: str) -> Optional[str]:
    """
    Parse the critic's verdict from XML tags in the response.
    Supports both <verdict> (new) and <judgment> (legacy) tags for backward compatibility.
    
    Args:
        response (str): The critic's response text
        
    Returns:
        Optional[str]: The verdict value ("NULL REJECTED", "NULL NOT REJECTED", 
                      "NOT TESTABLE") or None if not found. Legacy values are automatically converted.
    """
    if not response or not isinstance(response, str):
        return None
    
    # Look for verdict XML tags (new format) first, then judgment (legacy) for backward compatibility
    verdict_pattern = r'<verdict>(.*?)</verdict>'
    judgment_pattern = r'<judgment>(.*?)</judgment>'
    
    # Try new format first
    matches = re.findall(verdict_pattern, response, re.IGNORECASE | re.DOTALL)
    if not matches:
        # Fall back to legacy format
        matches = re.findall(judgment_pattern, response, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Get the last match (in case there are multiple)
        verdict = matches[-1].strip().upper()
        
        # Validate the verdict value (support both new statistical terminology and legacy values for backward compatibility)
        valid_verdicts = {
            # New statistical terminology
            "NULL REJECTED",
            "NULL NOT REJECTED",
            "NOT TESTABLE",
            # Legacy values (for backward compatibility)
            "HYPOTHESIS VERIFIED",
            "HYPOTHESIS NOT VERIFIED", 
            "NOT APPLICABLE"
        }
        
        if verdict in valid_verdicts:
            # Map legacy values to new terminology for consistency
            legacy_to_new = {
                "HYPOTHESIS VERIFIED": "NULL REJECTED",
                "HYPOTHESIS NOT VERIFIED": "NULL NOT REJECTED",
                "NOT APPLICABLE": "NOT TESTABLE"
            }
            return legacy_to_new.get(verdict, verdict)
    
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
    Check if the response contains a verdict XML tag.
    Supports both <verdict> (new) and <judgment> (legacy) tags.
    
    Args:
        response (str): The response text
        
    Returns:
        bool: True if verdict XML tag is found, False otherwise
    """
    return parse_critic_judgment(response) is not None
