import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 4,3
from matplotlib.animation import FuncAnimation
import csv

# opening the CSV file
with open('pendulum_by_Dyna_Q.csv', mode ='r') as file:
  csvFile = csv.reader(file)
  pend_data = list(csvFile)

fig, ax = plt.subplots() # create a figure with an axes

ax.axis([-2,2,-2,2]) # set the axes limits

ax.set_aspect("equal") # set equal aspect such that the circle is not shown as ellipse

# create a line in the axes
point, = ax.plot([0,-np.array(pend_data[0][1]).astype(np.float64)],[0,np.array(pend_data[0][0]).astype(np.float64)], marker="o") # starting point in first image

# Updating function, to be repeatedly called by the animation
def update(idx):
    # obtain point coordinates 
    point.set_data([0, -np.array(pend_data[int(idx)][1]).astype(np.float64)], [0, np.array(pend_data[int(idx)][0]).astype(np.float64)])
    return point

# create animation with 30ms interval, which is repeated,
ani = FuncAnimation(fig, update, interval=30, frames=np.linspace(0,len(pend_data),len(pend_data)+1,endpoint=False))

plt.show()