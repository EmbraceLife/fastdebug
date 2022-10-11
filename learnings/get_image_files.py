
files_test = get_image_files(path/"test")

def get_image_files(path, recurse=True, folders=None):====================================(0)       
    "Get image files in `path` recursively, only in `folders`, if specified."=============(1)       
    return get_files(path, extensions=image_extensions, recurse=recurse, folders=folders)=(2)       
                                                                                                                                                        (3)
