"""
Used for manipulating files and directories
"""

import os


# Manipulating directories
class FileManipulation():
    """
    Used to create a folder
    input: dir_name
    Creates a new directory of name dir_name
    """
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def remove_files(self):
        """
            Used to remove files inside folder
        """
        for files in os.listdir(self.dir_name):
            os.remove(os.path.join(self.dir_name, files))

    def make_directory_if_not_exists(self):
        """
            Used to create folder
        """
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        # else:
        #     os.remove(self.dir_name)
            # self.remove_files()
