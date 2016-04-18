import numpy as np
from PIL import Image
import os
import nibabel as nib
import time
import cv2
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab

import matplotlib.pyplot as plt

"""
img = nib.load('k220-T1_defaced.nii')

image_data = img.get_data()


for i in range(image_data.shape[2]):
    image_obj = Image.fromarray(image_data[:, :, i].astype(np.uint8))
    image_obj.show()
    time.sleep(0.1)

for i in range(image_data.shape[2]):
    plt.figure("Slice")
    plt.imshow(image_data[:, :, i], cmap='Greys_r')
    plt.show(block=False)
    #time.sleep(0.05)
    plt.clf()
"""
global data_path
data_path = 'data/k230-T1_defaced.nii'


class Tumor:

    def __init__(self):
        self.data_path = data_path
        self.img = nib.load(self.data_path)
        self.data = self.img.get_data()

    def animate_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        def animate():
            tstart = time.time()                   # for profiling
            img = nib.load(data_path)
            data = img.get_data()
            im = plt.imshow(self.data[:, :, 0], cmap='Greys_r')

            for i in range(data.shape[2]):
                im.set_data(data[:, :, i])
                fig.canvas.draw()                         # redraw the canvas
            # print 'FPS:' , 200/(time.time()-tstart)


        win = fig.canvas.manager.window
        fig.canvas.manager.window.after(100, animate)
        plt.show()

    def show_slice(self, i):
        data_slice = self.data[:, :, i].astype(np.uint8)
        equ = cv2.equalizeHist(data_slice)
        image_obj = Image.fromarray(equ)
        image_obj.show()
        img_orig = Image.fromarray(data_slice)
        image_obj.show()
        print equ.max(), data_slice.max()


tumor = Tumor()
tumor.show_slice(50)
tumor.animate_data()
# print image_data.shape[2]

# print img
