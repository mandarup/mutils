from __future__ import absolute_import, division, print_function

import os, shutil

def clean(path):
    """Remove all files from folder
    path = '/path/to/folder'
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
