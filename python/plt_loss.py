import csv

import matplotlib.pyplot as plt

csv_path = '/home/tomokimori/CLionProjects/3dreconGPU/python/loss.csv'

rows = []
with open(csv_path) as f:
    reader = csv.reader(f)
    rows = [row for row in reader]
data = [float(x) for x in rows[0][1:-1]]
plt.plot(data)
# plt.savefig('loss.png')
plt.show()
