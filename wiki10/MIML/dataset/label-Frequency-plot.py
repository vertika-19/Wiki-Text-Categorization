import numpy as np 
import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv("wikilabel_stats.csv", header=1)
# print(data)
plt.xlabel('Label Index')
plt.ylabel('No of document having this label')
plt.xticks( np.arange(0, 200, 50) )

freq = list(data['total_freq_count_for_label'])[:200]
plt.plot( freq )
plt.show()