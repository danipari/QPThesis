import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# count = 0
# trajectory = dict()
# # Strips the newline character
# with open("anOrbit.dat") as file_in:
#     for line in file_in:
#         lineSplit = line.split()
#         state = np.array([float(i) for i in lineSplit[1:]])
#         trajectory.update({lineSplit[0]: state})

# X = [trajectory[i][0] for i in trajectory]
# Y = [trajectory[i][1] for i in trajectory]
# Z = [trajectory[i][2] for i in trajectory]

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(X,Y,Z,'blue')
# plt.show()

# Method to draw a torus
def printTorus( torus ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for key in list(torus.keys()):
        circle = np.array(torus[key])
        if key == list(torus.keys())[0]:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'red')
        elif key == list(torus.keys())[-1]:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        elif key in np.linspace(0, 1, N1 + 1):
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'green')
        else:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'black')
    plt.show()

def readTorusDat(fileName):
    torus = dict()
    with open(fileName) as file_in:
        for line in file_in:
            lineSplit = line.split()
            time = float(lineSplit[0])
            angle = float(lineSplit[1])
            state = np.array([float(i) for i in lineSplit[2:]])

            if angle == 0:
                torus.update({time: []})

            torus[time].append(state)
    return torus

N1 =  10
a = readTorusDat("aTorus.dat")
printTorus(a)