import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

sys.path.append('../')
from nelder_mead.nelder_mead import NelderMead

class HimmelblauNelderMead(NelderMead):
    def buildSimplexPoints(self):
        x0 = np.array([-3.0, -4.0])
        x1 = np.array([-2.0, -2.0])
        x2 = np.array([0.0, -2.0])
        self.simplex = np.vstack((x0, x1, x2))

def himmelblau(f_variables):
    x = f_variables[0]
    y = f_variables[1]
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b

def background(ax, dimension):
    x = np.linspace(-dimension, dimension, 1000)
    y = np.linspace(-dimension, dimension, 1000)
    mesh = np.meshgrid(x, y)
    zz = himmelblau(mesh)
    xmin, xmax, ymin, ymax = (-dimension, dimension, -dimension, dimension)

    return ax.imshow(zz, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='jet', norm=colors.LogNorm(vmin=zz.min(), vmax=zz.max()))

def getSimplexAxis(simplex):
    x = np.append(simplex[:,0], simplex[0,0])
    y = np.append(simplex[:,1], simplex[0,1])
    return x,y

def plotOptimization():
    f_variables = np.array([0.0, -2.0])
    nelder = HimmelblauNelderMead(f_variables, use_shrink=True)

    fig, ax = plt.subplots(figsize=(5,5))
    dimension = 6
    ax.set_xlim(-dimension, dimension)
    ax.set_ylim(-dimension, dimension)

    steps = 30
    images = []
    n = 0

    simplex_copy = nelder.simplex.copy()

    for step in range(0, steps):
        if not np.allclose(nelder.simplex, simplex_copy):
            image1 = background(ax, dimension)

            x,y = getSimplexAxis(nelder.simplex)
            image2, = ax.plot(x, y, color="k", marker="o", markersize=4.0)

            if (n == 0):
                n += 1
                image3 = plt.colorbar(image1, ax=ax)
                plt.xlabel("$x_1$")
                plt.ylabel("$x_2$")

            simplex_copy = nelder.simplex.copy()
            images.append([image1, image2])

        f_value = himmelblau(f_variables)
        nelder.run(f_value)

    ani = animation.ArtistAnimation(fig, images, repeat=False, interval=600)
    ani.save('./Gifs/himmelblau.gif', writer='imagemagick', fps=2)
    plt.show()

if __name__ == '__main__':

    plotOptimization()
