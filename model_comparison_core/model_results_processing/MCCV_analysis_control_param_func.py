import os
import glob
import pandas as pd
import numpy as np
# Import the process_file function to ensure consistent metric extraction
from MCCV_analysis_grouping_base import process_file

def find_control_and_test_paths(main_dir, control_params):
    """
    Find control path and test paths based on control_params and directories in main_dir.
    
    Parameters:
    -----------
    main_dir : str
        Path to the main directory containing model result directories
    control_params : dict
        Dictionary of parameters to identify the control directory
        Special values:
        - {'model': 'best'} : Select the model with best overall performance
        - {'model': 'best per score'} : Select per-metric best models in analysis phase
        - {'model': 'directory_name'} : Select a specific directory by name
        - {param1: value1, param2: value2, ...} : Select by parameter values
        
    Returns:
    --------
    tuple
        (control_path, test_paths)
    """
    control_path = ""
    test_paths = []
    
    # Check if main_dir exists
    if not os.path.exists(main_dir) or not os.path.isdir(main_dir):
        print(f"Error: main_dir does not exist or is not a directory: {main_dir}")
        return control_path, test_paths
    
    # List all directories in main_dir
    subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    print(f"Found {len(subdirs)} subdirectories in main_dir")
    
    if not subdirs:
        print("Error: No subdirectories found in main_dir!")
        return control_path, test_paths
    
    # Check if control_params contains a 'model' key that directly specifies the control directory
    if 'model' in control_params:
        if control_params['model'] == 'best':
            print("\nSpecial case: Selecting the model with the best performance as control")
            control_path = find_best_performing_model(subdirs)
            # If metrics-based selection failed, check for a traditional control
            if not control_path:
                print("Metrics-based selection failed, checking for a directory with ';control' suffix")
                control_suffix_dirs = [d for d in subdirs if ";control" in os.path.basename(d)]
                if control_suffix_dirs:
                    control_path = control_suffix_dirs[0]
                    print(f"Found control directory with suffix: {os.path.basename(control_path)}")
                else:
                    # Ultimate fallback - first directory with CSV files
                    for d in subdirs:
                        csv_files = glob.glob(os.path.join(d, '*.csv'))
                        if csv_files:
                            control_path = d
                            print(f"Falling back to first directory with CSV files: {os.path.basename(control_path)}")
                            break
                    
                    # If all else fails, use the first directory
                    if not control_path and subdirs:
                        control_path = subdirs[0]
                        print(f"Using first directory as control: {os.path.basename(control_path)}")
        elif control_params['model'] == 'best per score':
            print("\nSpecial case: Will select best model per metric during analysis")
            # For this special case, we just return all directories without designating a specific control
            # The actual control selection happens during the analysis for each metric
            
            # Return all directories as test_paths
            test_paths = subdirs
            
            # We'll set a special control path to signal this mode
            control_path = "BEST_PER_SCORE"
            
            # In this mode, we're done - return early
            return control_path, test_paths
        else:
            # Regular case: Select by directory name
            control_name = control_params['model']
            print(f"\nSelecting control directory matching name: {control_name}")
            control_path = find_control_by_model_name(subdirs, control_name)
    else:
        # Select by parameters
        print(f"\nSelecting control directory matching parameters: {control_params}")
        control_path = find_control_by_parameters(subdirs, control_params)
    
    if not control_path:
        print("Error: Could not find a control directory!")
        return "", []
    
    # All other directories are test paths
    test_paths = [d for d in subdirs if d != control_path]
    
    print(f"Selected control: {os.path.basename(control_path)}")
    print(f"Found {len(test_paths)} test directories")
    
    return control_path, test_paths

def find_best_performing_model(subdirs):
    """
    Find the directory with the model that has the lowest Distance*Dampening/F1 score.
    Uses the same process_file function as the main MCCV analysis to ensure metrics are consistent.
    
    Parameters:
    -----------
    subdirs : list
        List of subdirectory paths to search
        
    Returns:
    --------
    str
        Path to the directory with the best performing model
    """
    print("\nLooking for the best performing model (lowest Distance*Dampening/F1 score)...")
    
    # Dictionary to store metric averages for each directory
    dir_metrics = {}
    
    for dir_path in subdirs:
        # Get all CSV files in this directory
        csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
        if not csv_files:
            print(f"  Skipping directory with no CSV files: {os.path.basename(dir_path)}")
            continue  # Skip directories with no CSV files
        
        print(f"\n  Processing directory: {os.path.basename(dir_path)}")
        print(f"  Found {len(csv_files)} CSV files")
        
        # Initialize metrics for this directory
        f1_scores = []
        distance_scores = []
        dampening_scores = []
        
        # Process each CSV file using the same function as the main MCCV analysis
        for csv_file in csv_files:
            try:
                print(f"    Processing file: {os.path.basename(csv_file)}")
                
                # Use the process_file function from MCCV_analysis_grouping_base
                # This ensures metrics are calculated exactly the same way
                try:
                    # Get metrics using the official process_file function
                    result = process_file(csv_file)
                    
                    # Check that we got valid metrics
                    if result and 'f1' in result and 'score' in result and 'freeze' in result:
                        f1 = result['f1']
                        distance = result['score']
                        dampening = result['freeze']
                        
                        # Verify metrics are valid values
                        if f1 > 0 and distance >= 0 and dampening >= 0:
                            f1_scores.append(f1)
                            distance_scores.append(distance)
                            dampening_scores.append(dampening)
                            print(f"      ✓ Extracted metrics using process_file: F1={f1:.3f}, Distance={distance:.3f}, Dampening={dampening:.3f}")
                        else:
                            print(f"      ✗ Invalid metric values: F1={f1}, Distance={distance}, Dampening={dampening}")
                    else:
                        print(f"      ✗ Missing required metrics in result: {result}")
                        
                except Exception as e:
                    print(f"      ✗ Error processing file with process_file: {str(e)}")
                    
            except Exception as e:
                print(f"      ✗ Error processing file: {str(e)}")
                continue
        
        # Calculate averages if we have data
        if f1_scores and distance_scores and dampening_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_distance = sum(distance_scores) / len(distance_scores)
            avg_dampening = sum(dampening_scores) / len(dampening_scores)
            
            # Calculate the combined score (Distance*Dampening/F1)
            if avg_f1 > 0:  # Avoid division by zero
                combined_score = (avg_distance * avg_dampening) / avg_f1
                dir_metrics[dir_path] = {
                    'F1': avg_f1,
                    'Distance': avg_distance,
                    'Dampening': avg_dampening,
                    'Combined': combined_score,
                    'NumFiles': len(f1_scores)
                }
                print(f"  ✓ Directory summary ({len(f1_scores)}/{len(csv_files)} files processed):")
                print(f"    F1: {avg_f1:.3f}, Distance: {avg_distance:.3f}, Dampening: {avg_dampening:.3f}")
                print(f"    Combined Score (Distance*Dampening/F1): {combined_score:.3f}")
            else:
                print(f"  ✗ Skipping directory due to zero F1 score: {os.path.basename(dir_path)}")
        else:
            print(f"  ✗ Could not extract valid metrics from any files in: {os.path.basename(dir_path)}")
    
    # Find the directory with the lowest combined score
    print(f"\nFound metrics for {len(dir_metrics)} directories")
    
    if dir_metrics:
        best_dir = min(dir_metrics.keys(), key=lambda k: dir_metrics[k]['Combined'])
        print(f"\n✓ Best performing model found: {os.path.basename(best_dir)}")
        print(f"  F1: {dir_metrics[best_dir]['F1']:.3f}")
        print(f"  Distance: {dir_metrics[best_dir]['Distance']:.3f}")
        print(f"  Dampening: {dir_metrics[best_dir]['Dampening']:.3f}")
        print(f"  Combined Score: {dir_metrics[best_dir]['Combined']:.3f}")
        print(f"  Files processed: {dir_metrics[best_dir]['NumFiles']}")
        return best_dir
    else:
        print("\n⚠ Warning: Could not calculate metrics for any directories")
        return ""

def find_control_by_model_name(subdirs, control_model_name):
    """
    Find control directory by exact or partial model name match.
    
    Parameters:
    -----------
    subdirs : list
        List of subdirectory paths to search
    control_model_name : str
        Model name to match
        
    Returns:
    --------
    str
        Path to the matched control directory, or first directory if no match
    """
    print(f"\nLooking for control directory with exact name: {control_model_name}")
    
    # Find the directory that matches the specified model name
    control_dirs = [d for d in subdirs if os.path.basename(d) == control_model_name]
    if control_dirs:
        control_path = control_dirs[0]
        print(f"Found control directory by exact name: {control_path}")
        return control_path
    
    print(f"Warning: No directory with name '{control_model_name}' found!")
    # Try a more flexible match (case-insensitive, partial match)
    control_dirs = [d for d in subdirs if control_model_name.lower() in os.path.basename(d).lower()]
    if control_dirs:
        control_path = control_dirs[0]
        print(f"Found control directory by partial match: {control_path}")
        return control_path
    
    print("Warning: No directory matching the specified model name found!")
    # Fall back to first directory
    control_path = subdirs[0]
    print(f"Using {control_path} as default control path (first directory)")
    return control_path

def find_control_by_parameters(subdirs, control_params):
    """
    Find control directory by matching parameters in directory names.
    
    Parameters:
    -----------
    subdirs : list
        List of subdirectory paths to search
    control_params : dict
        Dictionary of parameters to match
        
    Returns:
    --------
    str
        Path to the matched control directory, or first directory if no match
    """
    # Function to check if a directory name matches control parameters
    def matches_control_params(dir_name):
        dir_basename = os.path.basename(dir_name)
        print(f"Checking directory: {dir_basename}")
        
        # Temporarily remove any ";control" suffix for parameter extraction
        if ";control" in dir_basename:
            dir_basename = dir_basename.replace(";control", "")
            
        # Parse directory name into parameters
        params = {}
        param_pairs = dir_basename.split(';')
        
        for pair in param_pairs:
            if '-' in pair:
                category, value = pair.split('-', 1)  # Split on first hyphen only
                
                # Handle numeric parameters (including those with decimal points)
                if category == 'geo_params':
                    if value.replace('.', '', 1).isdigit():  # Handles both integers and floats
                        # Convert to same type as in control_params for comparison
                        if isinstance(control_params.get(category), int):
                            # If control parameter is integer, convert to int (removing decimal)
                            try:
                                value = int(float(value))
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(control_params.get(category), float):
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                pass
                
                params[category] = value
        
        # Debug: Print extracted parameters
        print(f"  Extracted parameters: {params}")
        print(f"  Control parameters: {control_params}")
        
        # Check if all control parameters are present with matching values
        for key, value in control_params.items():
            if key not in params:
                print(f"  Missing parameter: {key}")
                return False
            
            # Compare values (convert to string for consistent comparison)
            ctrl_value = str(value)
            dir_value = str(params[key])
            
            if ctrl_value != dir_value:
                print(f"  Parameter mismatch: {key} = {dir_value}, control = {ctrl_value}")
                return False
        
        print(f"  ✓ MATCH: This directory matches all control parameters")
        return True
    
    # Find control directory based on control_params
    print("\nSearching for control directory matching parameters:")
    for k, v in control_params.items():
        print(f"  {k}: {v}")
    
    control_dirs = [d for d in subdirs if matches_control_params(d)]
    if control_dirs:
        control_path = control_dirs[0]  # Use the first matching directory
        print(f"\nControl path set to: {control_path}")
        return control_path
    
    print("\nWarning: No directory matching control parameters!")
    # Check if there's a directory with ";control" suffix as fallback
    control_suffix_dirs = [d for d in subdirs if ";control" in os.path.basename(d)]
    if control_suffix_dirs:
        control_path = control_suffix_dirs[0]
        print(f"Falling back to directory with ;control suffix: {control_path}")
        return control_path
    
    # If no control directory found but directories exist, use the first one as control
    control_path = subdirs[0]
    print(f"Using {control_path} as default control path (first directory)")
    return control_path
