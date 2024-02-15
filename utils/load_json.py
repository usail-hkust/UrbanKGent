import json
import os
import shutil
def load_ndjson_to_array(file):
    data = []
    try:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data


def load_ndjson_to_dict(file):
    data = {}
    try:
        with open(file, 'r') as f:
            for line in f:
                data.update(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data


def load_ndjson(file, return_type='array'):
    if return_type == 'array':
        return load_ndjson_to_array(file)
    elif return_type == 'dict':
        return load_ndjson_to_dict(file)
    else:
        raise RuntimeError('Unknown return_type: %s' % return_type)

def delete_all_subdirectories(directory):
    try:
        for root, subdirs, files in os.walk(directory, topdown=False):
            for subdir in subdirs:
                subdirectory_path = os.path.join(root, subdir)
                shutil.rmtree(subdirectory_path)
                print(f"Deleted subdirectory: {subdirectory_path}")
        print(f"All subdirectories in '{directory}' have been deleted.")
    except Exception as e:
        print(f"Error deleting subdirectories: {e}")