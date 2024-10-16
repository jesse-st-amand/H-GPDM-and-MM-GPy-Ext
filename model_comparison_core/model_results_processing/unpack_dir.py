import os
import shutil

# Source directory
source_dir = r"C:\Users\Jesse\Documents\Python\GPy\HGPLVM_output_repository\model_summaries\RNN_params_Bayesian_BM\f1_dist_msad_Bimanual 3D_Bayesian_RNN"

# Get the parent directory and the name of the source directory
parent_dir = os.path.dirname(source_dir)
source_dir_name = os.path.basename(source_dir)

# Create the new 'copy_' directory name
copy_dir_name = f"unpacked_{source_dir_name}"
copy_dir = os.path.join(parent_dir, copy_dir_name)

# Create the 'copy_' directory
if not os.path.exists(copy_dir):
    os.makedirs(copy_dir)

# Copy the entire source directory to the new 'copy_' directory
shutil.copytree(source_dir, copy_dir, dirs_exist_ok=True)

# Now set the copy_dir as our working directory
working_dir = copy_dir

# Get all subdirectories in the working directory
sub_dirs = [f.path for f in os.scandir(working_dir) if f.is_dir()]

for dir_path in sub_dirs:
    # Get all items in the subdirectory
    for root, _, files in os.walk(dir_path):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(working_dir, file)
            # Move the file to the root of the working directory
            shutil.move(src_path, dest_path)

    # Remove the now-empty subdirectory
    shutil.rmtree(dir_path)

print(f"Finished unpacking all folders to {working_dir}")
print("Original subdirectories have been removed.")