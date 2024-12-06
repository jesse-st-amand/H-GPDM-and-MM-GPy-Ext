import os
import shutil
import re

def organize_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Get all CSV files from the source folder
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    
    # Dictionary to store files by their dict number
    files_by_dict = {}
    
    # Process each file
    for file in csv_files:
        # Extract dict number using regex
        match = re.search(r'dict_(\d+)', file)
        if match:
            dict_num = int(match.group(1))
            files_by_dict[dict_num] = files_by_dict.get(dict_num, []) + [file]
    
    # Group files by ranges of 50
    for dict_num in files_by_dict.keys():
        # Calculate which group this dict number belongs to
        group_num = dict_num // 50
        group_start = group_num * 50
        group_end = group_start + 49
        
        # Create folder name based on the range
        folder_name = f"dict_{group_start}_to_{group_end}"
        group_folder = os.path.join(destination_folder, folder_name)
        
        # Create the group folder if it doesn't exist
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)
        
        # Move files to their respective folders
        for file in files_by_dict[dict_num]:
            source_path = os.path.join(source_folder, file)
            dest_path = os.path.join(group_folder, file)
            shutil.copy2(source_path, dest_path)  # Using copy2 to preserve metadata

def main():
    # Replace these paths with your actual source and destination folders
    source_folder = r"C:\Users\Jesse\Documents\Python\HGP_concise\HGPLVM_output_repository\model_summaries\MCCV\GPDM_MCCV_BM_IC_testing\GPDMM_testing_inits_MCCV_f1_dist_msad_Bimanual 3D - Copy"
    destination_folder = r"C:\Users\Jesse\Documents\Python\HGP_concise\HGPLVM_output_repository\model_summaries\MCCV\GPDM_MCCV_BM_IC_testing\New folder"
    
    try:
        organize_files(source_folder, destination_folder)
        print("Files have been successfully organized!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()