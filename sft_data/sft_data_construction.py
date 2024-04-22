import json
import os
import shutil
import os
import glob

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data
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


if __name__ == "__main__":
    # 指定要搜索的文件夹路径
    folder_path = '../prompt/fine_tune/'

    # 使用glob模式匹配来找到所有的.json文件
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    instruction_set = []

    for json_file in json_files:

        instructions  = load_ndjson(json_file)

        for i in range(len(instructions)):

            instructions_dict_index = {}

            instruction = ''
            if len(instructions[i]['prompt_completion']) ==1:
                continue
            else:
                for j in range(len(instructions[i]['prompt_completion']['messages'])):
                    instruction = instruction + str(instructions[i]['prompt_completion']['messages'][j]['content'])
            output = instructions[i]['response']

            instructions_dict_index['instruction'] = instruction
            instructions_dict_index['input'] = ''
            instructions_dict_index['output'] = output

            instruction_set.append(instructions_dict_index)

        print('done')


    print(len(instruction_set))
    dump_json(instruction_set, 'instructions_example.json')


