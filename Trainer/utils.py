# TODO : write a structure grid generator

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata

class Plot2D:
    def Contour2D(nodecoords,sol,savefig=False,figname=''):
        fig, ax = plt.subplots(constrained_layout=True)
        x,y=nodecoords[:,0],nodecoords[:,1]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
        Z = griddata(nodecoords, sol, (X, Y))
        #X,Y=np.meshgrid(x,y)
        #Z=griddata((x,y),sol,(X,Y),method='linear')
        cs=plt.contourf(X,Y,Z,cmap=plt.cm.hsv,levels=200)
        #cs=plt.contourf(X,Y,Z,cmap=plt.cm.viridis,levels=200,antialiased=True,extend='both')
        #cs=plt.tricontourf(x,y,sol,levels=20,cmap="jet")
        fig.colorbar(cs)
        ax.axis('equal')

        if savefig:
            if len(figname)>4:
                fig.savefig(figname,dpi=300,bbox_inches='tight')
                print('save results to ',figname)
            else:
                fig.savefig('result.jpg',dpi=300,bbox_inches='tight')
                print('save result to result.jpg')

#Class StrgirdEngine: