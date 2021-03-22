
import os

import numpy as np


path = '4K_120'

classes = {'HA':0, 'HB':1, 'HN':2, 'MA':3, 'MB':4, 'MN':5}

clss = os.listdir(path)

print(clss)

for c in clss:

    key = c.split('4K_120')[0]
    print(key, classes[key])

    group_path = os.path.join(path, c)
    print(group_path)

    data = os.listdir(group_path)
    print(data)

    # From each just get one picture.
