import numpy as np
import os

sum = 0
filelist = os.listdir("./seg")
for file in filelist:
    if len(file)>10:
        print(file)
        sum += 1
print(sum)

