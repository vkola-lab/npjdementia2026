#!/usr/bin/env python3
"""
Script to combine batch_old.csv and batch.csv, then reformat the data
so that each hypothesis has its citations in separate rows.
"""

import csv
import sys

def reformat_citations(input_file1, input_file2, output_file):
    """
    Combine two CSV files and reformat citations.
    
    Format:
    hypothesis, citation1, link1
    , citation2, link2
    , citation3, link3
    """
    
    # Read both CSV files
    rows = []
    
    # Read batch_old.csv (skip header)
    with open(input_file1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Read batch.csv (skip header)
    with open(input_file2, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Write reformatted output
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['hypothesis_number', 'hypothesis', 'citation', 'link'])
        
        # Process each row
        hypothesis_num = 1
        for row in rows:
            hypothesis = row.get('hypothesis', '').strip()
            
            # Get citations and links
            citations = [
                row.get('Citation 1', '').strip(),
                row.get('Citation 2', '').strip(),
                row.get('Citation 3', '').strip()
            ]
            links = [
                row.get('Link 1', '').strip(),
                row.get('Link 2', '').strip(),
                row.get('Link 3', '').strip()
            ]
            
            # Write first citation with hypothesis number and text (only in first row)
            if citations[0] or links[0]:
                writer.writerow([hypothesis_num, hypothesis, citations[0], links[0]])
            
            # Write subsequent citations with empty hypothesis_number and hypothesis columns
            for i in range(1, 3):
                if citations[i] or links[i]:
                    writer.writerow(['', '', citations[i], links[i]])
            
            # Increment hypothesis number for next hypothesis
            hypothesis_num += 1

if __name__ == '__main__':
    input_file1 = 'batch_old.csv'
    input_file2 = 'batch.csv'
    output_file = 'reformatted_citations.csv'
    
    reformat_citations(input_file1, input_file2, output_file)
    print(f"Reformatted citations written to {output_file}")

