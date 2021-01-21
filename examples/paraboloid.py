import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

sys.path.append('../')
from nelder_mead.nelder_mead import NelderMead

class ParaboloidNelderMead(NelderMead):
    def buildSimplexPoints(self):
        x0 = np.array([1.0, 1.0])
        x1 = np.array([2.5, 1.0])
        x2 = np.array([1.5, 2.0])
        self.simplex = np.vstack((x0, x1, x2))

def paraboloid(f_variables):
    alpha = 2.5
    beta = 2.5
    return np.sqrt((f_variables[0]-alpha)**2 + (f_variables[1]-beta)**2)

def background(ax, dimension):
    x = np.linspace(0, dimension, 1000)
    y = np.linspace(0, dimension, 1000)
    mesh = np.meshgrid(x, y)
    zz = paraboloid(mesh)
    xmin, xmax, ymin, ymax = (0, dimension, 0, dimension)

    return ax.imshow(zz, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='jet', norm=colors.LogNorm(vmin=zz.min(), vmax=zz.max()))

def getSimplexAxis(simplex):
    x = np.append(simplex[:,0], simplex[0,0])
    y = np.append(simplex[:,1], simplex[0,1])
    return x,y

def plotOptimization():
    f_variables = np.array([1.5, 2.0])
    nelder = ParaboloidNelderMead(f_variables)

    fig, ax = plt.subplots(figsize=(5,5))
    dimension = 5
    ax.set_xlim(0, dimension)
    ax.set_ylim(0, dimension)

    steps = 30
    images = []
    n = 0

    simplex_copy = nelder.simplex.copy()

    for step in range(0, steps):
        if not np.allclose(nelder.simplex, simplex_copy):
            image1 = background(ax, dimension)

            x,y = getSimplexAxis(nelder.simplex)
            image2, = ax.plot(x, y, color="k", marker="o",  markersize=4.0)

            if (n == 0):
                n += 1
                image3 = plt.colorbar(image1, ax=ax)
                plt.xlabel("$x_1$")
                plt.ylabel("$x_2$")

            simplex_copy = nelder.simplex.copy()
            images.append([image1, image2])

        f_value = paraboloid(f_variables)
        nelder.run(f_value)

    ani = animation.ArtistAnimation(fig, images, repeat=False, interval=600)
    ani.save('./Gifs/paraboloid.gif', writer='imagemagick', fps=2)
    plt.show()

if __name__ == '__main__':

    plotOptimization()
