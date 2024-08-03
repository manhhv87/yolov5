

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-3
Labelling new Dataset in YOLO format
File: creating-train-and-test-txt-files.py
"""


# Creating files train.txt and test.txt
# for training in Darknet framework
#
# Algorithm:
# Setting up full paths --> List of paths -->
# --> Extracting 15% of paths to save into test.txt file -->
# --> Writing paths into train and test txt files
#
# Result:
# Files train.txt and test.txt with full paths to images


# Importing needed library
import os


"""
Start of:
Setting up full path to directory with labelled images
"""

# Full or absolute path to the folder with images
# Find it with Py file getting-full-path.py
# Pay attention! If you're using Windows, yours path might looks like:
# r'C:\Users\my_name\Downloads\video-to-annotate'
# or:
# 'C:\\Users\\my_name\\Downloads\\video-to-annotate'
full_path_to_images = '/content/yolov5/datasets'

"""
End of:
Setting up full path to directory with labelled images
"""


"""
Start of:
Getting list of full paths to labelled images
"""

# Check point
# Getting the current directory
# print(os.getcwd())

# Changing the current directory
# to one with images
os.chdir(full_path_to_images)

# Check point
# Getting the current directory
# print(os.getcwd())

# Defining list to write paths in
p = []

# Using os.walk for going through all directories
# and files in them from the current directory
# Fullstop in os.walk('.') means the current directory
for current_dir, dirs, files in os.walk('.'):
    for subdir in dirs:
        for x in os.listdir(subdir):
            if x[-3:] == "jpg":
                path_to_images = os.path.join(subdir, x)
                path_to_save_into_txt_files = os.path.join(full_path_to_images, path_to_images)
                p.append(path_to_save_into_txt_files + '\n')


# Slicing first 15% of elements from the list
# to write into the val.txt file
p_val = p[:int(len(p) * 0.2)]

# Deleting from initial list first 15% of elements
p = p[int(len(p) * 0.15):]

"""
End of:
Getting list of full paths to labelled images
"""


"""
Start of:
Creating train.txt and test.txt files
"""

# Creating file train.txt and writing 85% of lines in it
with open('/content/yolov5/datasets/train.txt', 'w') as train_txt:
    # Going through all elements of the list
    for e in p:
        # Writing current path at the end of the file
        train_txt.write(e)

# Creating file val.txt and writing 15% of lines in it
with open('/content/yolov5/datasets/val.txt', 'w') as val_txt:
    # Going through all elements of the list
    for e in p_val:
        # Writing current path at the end of the file
        val_txt.write(e)

"""
End of:
Creating train.txt and val.txt files
"""
