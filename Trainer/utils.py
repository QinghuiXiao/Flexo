import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata

class Plot2D:
    def Quiver2D(nodecoords, sol, savefig=False, figname=''):
        fig, ax = plt.subplots(constrained_layout=True)
        x, y=nodecoords[:, 0],nodecoords[:, 1]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 20))
        P1 = sol[:, 0]
        P2 = sol[:, 1]
        magnitude = np.sqrt(P1**2 + P2**2)
        P1_normalized = P1 / magnitude
        P2_normalized = P2 / magnitude

        cs= plt.quiver(X, Y, P1_normalized, P2_normalized, magnitude, angles='xy', scale_units='xy', scale=1, cmap='jet')

        fig.colorbar(cs)
        ax.axis('equal')

        if savefig:
            if len(figname)>4:
                fig.savefig(figname,dpi=300,bbox_inches='tight')
                print('save results to ',figname)
            else:
                fig.savefig('result.jpg',dpi=300,bbox_inches='tight')
                print('save result to result.jpg')

class StrgridEngine:
    def __init__(self, dimension=2, grid_size=(30, 20)):
        self.dimension = dimension
        self.grid_size = grid_size

    def generate_structure_grid(self):
        # Generate a structured grid of points
        x_points = torch.linspace(0, 1, self.grid_size[0])
        y_points = torch.linspace(0, 1, self.grid_size[1])

        # Create grid using meshgrid
        grid_x, grid_y = torch.meshgrid(x_points, y_points)

        # Flatten the grid
        flattened_grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

        return flattened_grid
