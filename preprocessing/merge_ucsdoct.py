import os
import shutil

# Define the source paths
ucsdoct_folder = "./datasets/ucsdoct/"
train_folder = os.path.join(ucsdoct_folder, 'train')
test_folder = os.path.join(ucsdoct_folder, 'test')
merged_folder = './datasets/ucsdoct/'

# Create the merged folder if it doesn't exist
if not os.path.exists(merged_folder):
    os.makedirs(merged_folder)

# List of subdirectories (categories) to merge
categories = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Function to merge files from train and test folders into a single folder
def merge_folders(src_folder_1, src_folder_2, dest_folder, categories):
    for category in categories:
        src_folder_1_category = os.path.join(src_folder_1, category)
        src_folder_2_category = os.path.join(src_folder_2, category)
        dest_category_folder = os.path.join(dest_folder, category)

        # Create category folder in merged folder if it doesn't exist
        if not os.path.exists(dest_category_folder):
            os.makedirs(dest_category_folder)

        # Copy files from the train folder
        for filename in os.listdir(src_folder_1_category):
            src_file = os.path.join(src_folder_1_category, filename)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dest_category_folder)

        # Copy files from the test folder
        for filename in os.listdir(src_folder_2_category):
            src_file = os.path.join(src_folder_2_category, filename)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dest_category_folder)

# Merge the train and test folders into the merged folder
merge_folders(train_folder, test_folder, merged_folder, categories)

print(f"Train and test folders have been merged into {merged_folder}.")