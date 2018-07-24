import numpy as np
import pandas as pd

a = [0] * 3
b = [1] * 5
c = []
c.append(a)
c.append(a)

df4 = pd.DataFrame({'col1':[1,3],'col2':[2,4]},index=['a','b'])
s = 'a'
print(np.sum(b))
#print(b.mean(1))
