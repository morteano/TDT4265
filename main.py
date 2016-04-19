import pickle
import os.path
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
            im = plt.imshow(self.data[:, :, 0], cmap='gray')

            for i in range(data.shape[2]):
                im.set_data(cv2.fastNlMeansDenoising(data[:, :, i].astype(np.uint8), None, 10, 7, 21))
                fig.canvas.draw()                         # redraw the canvas

            # print 'FPS:' , 200/(time.time()-tstart)

        win = fig.canvas.manager.window
        fig.canvas.manager.window.after(100, animate)
        plt.show()

    def show_slice(self, i):
        data_slice = self.data[:, :, i].astype(np.uint8)
        equ = cv2.equalizeHist(data_slice)
        # image_obj = Image.fromarray(equ)
        # image_obj.show(title='equ')
        img_orig = Image.fromarray(data_slice)
        img_orig.show(title='orig')
        print equ.mean(), data_slice.max()

    def top_hat_slice(self, i):
        data_slice = self.data[:, :, i].astype(np.uint8)
        equ = cv2.equalizeHist(data_slice)
        blur = cv2.GaussianBlur(equ, (3, 3), 0)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        th = cv2.morphologyEx(data_slice, cv2.MORPH_TOPHAT, ker)
        image_blur = Image.fromarray(blur)
        image_blur.show(title='blur')
        image_top = Image.fromarray(th)
        image_top.show(title='top')

    def denoise(self):
        data = self.get_data()
        denoised = np.empty_like(data)
        print data.shape, denoised.shape
        for i in range(data.shape[2]):
            denoised[:, :, i] = cv2.fastNlMeansDenoising(data[:, :, i].astype(np.uint8), None, 10, 7, 21)
        # blur = cv2.GaussianBlur(test, (3, 3), 0)
        # equ = cv2.equalizeHist(test)
        # plt.imshow(test, cmap="gray")
        # plt.show()
        # plt.imshow(blur, cmap="gray")
        # plt.show()
        return denoised

    def get_data(self, slice=None):
        if slice:
            return self.data[:, :, slice].astype(np.uint8)
        else:
            return self.data

    def animate_scan(self, scan_data):
        for i in range(scan_data.shape[2]):
            cv2.imshow("test", scan_data[:, :, i].astype(np.uint8))
            cv2.waitKey(50)

    def animate_hard(self):
        data = self.get_data()
        for i in range(data.shape[2]):
            cv2.imshow("test", cv2.fastNlMeansDenoising(data[:, :, i].astype(np.uint8), None, 10, 7, 21))
            cv2.waitKey(50)

    def getDenoised(self):
        if os.path.isfile("denoiseImgs"):
            file = open("denoiseImgs", 'rb')
            denoised = pickle.load(file)
            file.close()
        else:
            denoised = self.denoise()
            file = open("denoiseImgs","wb")
            pickle.dump(denoised, file)
            file.close()
        return denoised

tumor = Tumor()
# tumor.show_slice(50)
#tumor.animate_data()
#tumor.top_hat_slice(50)
#tumor.denoise_test(50)
denoised = tumor.getDenoised()
tumor.animate_scan(denoised)
# tumor.animate_hard()
# print image_data.shape[2]

# print img