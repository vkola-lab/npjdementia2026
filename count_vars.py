#!/usr/bin/env python3
"""
Script to count the number of variables in test.json
"""

import json
import sys
import os

def count_variables(json_file_path):
    """
    Count the number of variables in the JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        int: Number of variables found
    """
    try:
        # Check if file exists
        if not os.path.exists(json_file_path):
            print(f"Error: File '{json_file_path}' not found.")
            return 0
            
        # Read and parse the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Count the number of top-level keys (variables)
        if isinstance(data, dict):
            var_count = len(data.keys())
            print(f"Number of variables in '{json_file_path}': {var_count}")
            return var_count
        else:
            print(f"Error: Expected JSON object, got {type(data)}")
            return 0
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0

def main():
    """Main function"""
    # Default path to test.json in the same directory as this script
    default_path = os.path.join(os.path.dirname(__file__), 'test.json')
    
    # Use command line argument if provided, otherwise use default
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = default_path
    
    print(f"Analyzing file: {json_file_path}")
    count_variables(json_file_path)

if __name__ == "__main__":
    main()




