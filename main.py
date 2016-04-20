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
#data_path = 'data/k230-T1_defaced.nii'
#data_path = 'data/k224-FLAIR_defaced.nii'
data_path = 'data/k230-FLAIR_defaced.nii'
#data_path = 'data/k211-T1_defaced.nii'


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
            orig = scan_data[:, :, i].astype(np.uint8)
            equ = cv2.equalizeHist(orig)
            ret, thresh1 = cv2.threshold(orig, 115, 255, cv2.THRESH_TOZERO_INV)
            equ = cv2.equalizeHist(thresh1)
            ret, thresh2 = cv2.threshold(equ, 200, 255, cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(thresh2, (5, 5), 0)
            harris = cv2.cornerHarris(thresh2, 2, 3, 0.04)
            inverted = 255 - equ
            reverted = 255 - thresh1

            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            kernel2 = np.ones((5, 5), np.uint8)
            opening2 = cv2.morphologyEx(unknown, cv2.MORPH_OPEN, kernel)

            cv2.imshow("test", opening2)
            cv2.waitKey(50)

    def animate_test(self, scan_data):
        for i in range(10):
            orig = scan_data[:, :, 70].astype(np.uint8)
            equ = cv2.equalizeHist(orig)
            inverted = 255 - equ
            ret, thresh1 = cv2.threshold(orig, 110+5*i, 255, cv2.THRESH_TOZERO_INV)
            reverted = 255 - thresh1
            cv2.imshow("test", thresh1)
            cv2.waitKey(1000)

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