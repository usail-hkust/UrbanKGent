import re

def str2triplet(str):

    matches = re.findall(r'\{(.*?)\}', str)
    if len(matches) == 0:
        return str
    return matches