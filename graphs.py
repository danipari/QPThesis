import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import roots_legendre
from matplotlib import cm

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
    for key in list(torus.keys()):
        circle = np.array(torus[key])
        if key == list(torus.keys())[0]:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        elif key == list(torus.keys())[-1]:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        elif key in np.linspace(0, 1, N1 + 1):
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        else:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')

def printTrajectory( X, Y, Z ):
        ax.plot(X, Y, Z, 'black')

def readTorusDat(fileName):
    torus = dict()
    with open(fileName) as file_in:
        for line in file_in:
            lineSplit = line.split()
            time = float(lineSplit[0])
            angle = float(lineSplit[1])
            state = np.array([float(i) for i in lineSplit[2:]])

            if angle == 0:
                torus.update({time: {}})

            torus[time].update({angle: state})
    return torus

def readTorusDatSurf(fileName):
    X = []
    Y = []
    Z = []
    with open(fileName) as file_in:
        for line in file_in:
            lineSplit = line.split()
            time = float(lineSplit[0])
            angle = float(lineSplit[1])
            state = np.array([float(i) for i in lineSplit[2:]])
            X.append(state[0])
            Y.append(state[1])
            Z.append(state[2])

    return X,Y,Z

def readTrajectory(fileName):
    X = []
    Y = []
    Z = []
    with open(fileName) as file_in:
        for line in file_in:
            lineSplit = line.split()
            state = np.array([float(i) for i in lineSplit[1:]])
            X.append(state[0])
            Y.append(state[1])
            Z.append(state[2])

    return X, Y, Z

def createGaussLegendreCollocationArray( N, m ):
    collocationaArray = []

    for intervalValue in np.linspace(0, 1, N+1):
        collocationaArray.append(intervalValue)
        # Break the last iteration before filling
        if intervalValue == 1.0: break
        offset = intervalValue
        for root in roots_legendre(m)[0]:
            collocationaArray.append(offset + (root / 2.0 + 0.5) / N) # Tranform to the interval [0,1]

    return collocationaArray

myTorus = readTorusDat("aTorus.dat")
def myfuncX(theta, phi):
    if phi == 1:
        phi = 0.0
    return myTorus[theta][phi][0]

def myfuncY(theta, phi):
    if phi == 1:
        phi = 0.0
    return myTorus[theta][phi][1]

def myfuncZ(theta, phi):
    if phi == 1:
        phi = 0.0
    return myTorus[theta][phi][2]

# theta = np.array(list(myTorus.keys()))
# phi =  np.concatenate([np.array(list(myTorus[0].keys())),[1]])
# theta, phi = np.meshgrid(theta, phi)

# vfuncX = np.vectorize(myfuncX)
# vfuncY = np.vectorize(myfuncY)
# vfuncZ = np.vectorize(myfuncZ)

# x = vfuncX(theta, phi)
# y = vfuncY(theta, phi)
# z = vfuncZ(theta, phi)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, color='b', shade=False, alpha = 0.8, edgecolor = 'k')

# X, Y, Z = readTrajectory("aTrajectory.dat")
# printTrajectory(X, Y, Z)
# plt.show()

def readOrbit(fileName):
    X = []
    Y = []
    Z = []
    with open(fileName) as file_in:
        for line in file_in:
            lineSplit = line.split()
            time = float(lineSplit[0])
            state = np.array([float(i) for i in lineSplit[1:]])
            X.append(state[0])
            Y.append(state[1])
            Z.append(state[2])

    return X, Y, Z

X, Y, Z = readOrbit("perOrbit.dat")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z)
plt.show()