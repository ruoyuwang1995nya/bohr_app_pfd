import os
from .constants import default_type_map
def get_element(file_path:str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File does not exist!")
    
    entries=os.path.basename(file_path).split('.',maxsplit=1)[0].split('_',maxsplit=1)
    if entries[0] not in default_type_map:
        raise NameError("Wrong element names!")
    return entries[0]

if __name__ == '__main__':
    with open("H.pp","w") as fp:
        fp.write("foo")
    print(get_element("H.pp"))