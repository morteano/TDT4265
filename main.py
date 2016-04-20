import pickle
import numpy as np
import os
import nibabel as nib
import cv2


global data_path
data_path = 'data/k225-T1_defaced.nii'
# data_path = 'data/k230_slicer_2_als-label.nii'
#data_path = 'data/k230-T1_defaced.nii'


class Tumor:

    def __init__(self):
        self.data_path = data_path
        self.img = nib.load(self.data_path)
        self.data = self.img.get_data()
        self.denoised = self.get_denoised()

    def show_slice(self, data, i):
        data_slice = data[:, :, i].astype(np.uint8)
        cv2.imshow("orig", data_slice)
        cv2.waitKey(0)


    def top_hat_slice(self, i):
        data_slice = self.data[:, :, i].astype(np.uint8)
        equ = cv2.equalizeHist(data_slice)
        blur = cv2.GaussianBlur(equ, (3, 3), 0)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        th = cv2.morphologyEx(data_slice, cv2.MORPH_TOPHAT, ker)
        cv2.imshow("blur", blur)
        cv2.waitKey(0)
        cv2.imshow("top hat", th)
        cv2.waitKey(0)

    def denoise(self):
        data = self.get_data()
        denoised = np.empty_like(data)
        for i in range(data.shape[2]):
            denoised[:, :, i] = cv2.fastNlMeansDenoising(data[:, :, i].astype(np.uint8), None, 10, 7, 21)
        return denoised

    def get_data(self, slice=None):
        if slice:
            return self.data[:, :, slice].astype(np.uint8)
        else:
            return self.data

    def animate_scan(self, scan_data):
        for i in range(scan_data.shape[2]):
            cv2.imshow("test", cv2.medianBlur(scan_data[:, :, i].astype(np.uint8), 5))
            cv2.waitKey(50)

    def get_denoised(self):
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
#tumor.show_slice(tumor.denoised, 73)
tumor.animate_scan(tumor.denoised)
tumor.animate_scan(tumor.data)
# tumor.animate_hard()
# print image_data.shape[2]

# print img