"""Make an MP4 file of the output of the LUC model of Mozambique
Judith Verstegen, 2017-07-04

"""

from matplotlib import animation
from matplotlib import colors as cls
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from pcraster import *
import parameters

##############
### inputs ###
##############

timesteps = parameters.getNrTimesteps()
samples = parameters.getNrSamples()
init_year = 1991
fn1 = 'urb-ave'
legend_loc = (1.01,0.25)

############
### MAIN ###
############

# open the raster and read out the data in a numpy array
wd = os.getcwd()
if (8 - len(fn1)) > 0: nr_zeros = 8 - len(fn1)
else:  nr_zeros = 0
in_fn = os.path.join(wd, fn1 + nr_zeros * '0' + '.001')
setclone(in_fn)
amap = readmap(in_fn)
data = pcr2numpy(amap, -999)
data = np.ma.masked_where(data<0, data)

# create the figure
f, axarr = plt.subplots(1)
plt.axis('off')
#norm = cls.Normalize(vmin=0, vmax=1)

# text boxes
title = axarr.text(0.05, 1.02, '', \
                   transform=axarr.transAxes)
year = axarr.text(0.70, 1.02, '', transform=axarr.transAxes)

# use imshow to plot the raster over time
# in two functions for the animation
def init_movie():
    im = axarr.imshow(data, cmap='gist_heat_r', animated=True)
    cb = plt.colorbar(im, spacing='proportional')
    return im, year, title

def animate(i):
    t = i + 1
    if t < 10:
        fn = in_fn[:-3] + '00' + str(t)
    else:
        fn = in_fn[:-3] + '0' + str(t)
    ##print fn
    amap = readmap(fn)
    data = pcr2numpy(amap, -999)
    data = np.ma.masked_where(data<0, data)
    im = axarr.imshow(data, cmap='gist_heat_r', animated=True)
    year.set_text('year = ' + str(init_year + t - 1))
    title.set_text('probability built-up')
    return im, year, title

im_ani = animation.FuncAnimation(f, animate, interval=400, \
                                   blit=False, frames = timesteps,\
                                   init_func=init_movie)
im_ani.save('movie_' + fn1 + '.mp4', dpi=300, \
            metadata={'artist':'Judith Verstegen'})
#plt.show()
