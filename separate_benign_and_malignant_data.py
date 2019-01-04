# place this script in the same hierarchy as "all_mias" directory

import os
import sys
import shutil

# path to all images
all_dataset_path = 'all_mias/'

# path to save benign images 
cleaned_dataset_path_b = 'dataset/benign/'

# path to save malignant images 
cleaned_dataset_path_m = 'dataset/malignant/'

# open 'filenames.txt'
# filename.txt is created using the information about the
# images from the MIAS database
print('Processing...')
with open('filenames.txt', 'r') as f:
    for line in f:
        if ' B ' in line:
            # source file from "all_mias"  directory
            src = all_dataset_path + str(line.split(' ')[0]) + '.pgm'

            # destination file for benign images
            dst = cleaned_dataset_path_b + str(line.split(' ')[0]) + '_benign.pgm'

            # copy benign images to 'dataset/b/'                                       
            shutil.copy2(src, dst)
                                                    
        if ' M ' in line:
            # source file from "all_mias"  directory
            src1 = all_dataset_path + str(line.split(' ')[0]) + '.pgm'

            # destination file for malignant images            
            dst1 = cleaned_dataset_path_m + str(line.split(' ')[0]) + '_malignant.pgm'

            # copy malignant images to 'dataset/m/'      
            shutil.copy2(src1, dst1)

print('Finished processing!')
