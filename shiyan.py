import numpy as np

arr=np.random.uniform(low=0.0, high=1.0, size=(1000, 1000))
tal = np.sum(arr[100:200, 100:200] > 0.5)
print(tal)
