#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#imgfileサイズ依存しないように改変
#original codeでは、予め（128,128.3）で格納しておく必要がある

import numpy as np
import glob
import argparse
import h5py

from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', '-i', required=True)
    parser.add_argument('--outpath', '-o', required=True)
    parser.add_argument('--trans', '-t', default='gray')
    args = parser.parse_args()
    img_rows =128
    img_cols =128

    finders = glob.glob(args.inpath+'/*')
    print(finders)
    imgs = []
    gimgs = []
    for finder in finders:
        files = glob.glob(finder+'/*')
        for imgfile in files:
            img = load_img(imgfile, target_size=(img_rows,img_cols))
            imgarray = img_to_array(img)
            imgs.append(imgarray)
            if args.trans=='gray':
                grayimg = load_img(imgfile, grayscale=True, target_size=(img_rows,img_cols))
                grayimgarray = img_to_array(grayimg)
                gimgs.append(grayimgarray)
            elif args.trans=='canny':
                img = image.load_img(imgfile, target_size=(img_rows,img_cols))
                img.save(imgfile)
                #grayimg = cv2.cvtColor(cv2.imread(imgfile, cv2.COLOR_BGR2GRAY)) #imgfileのサイズに依存しないように変更
                grayimg = cv2.imread(imgfile,0)                
                print("gray_canny_xy",grayimg.shape)
                gray_canny_xy = cv2.Canny(grayimg, 128,128 )
                print("gray_canny_xy",gray_canny_xy.shape)
                gray_canny_xy = cv2.bitwise_not(gray_canny_xy)
                print("gray_canny_xy",gray_canny_xy.shape)
                gimgs.append(gray_canny_xy.reshape(128,128,1))                
                    

    perm = np.random.permutation(len(imgs))
    vimgs = np.array(imgs)[perm]
    vgimgs = np.array(gimgs)[perm]
    #threshold = len(imgs)   #//10*9
    #vimgs = imgs[threshold:]
    #vgimgs = gimgs[threshold:]
    #imgs = imgs[:threshold]
    #gimgs = gimgs[:threshold]
    print('shapes')
    #print('gen imgs : ', imgs.shape)
    #print('raw imgs : ', gimgs.shape)
    print('val gen  : ', vimgs.shape)
    print('val raw  : ', vgimgs.shape)

    outh5 = h5py.File(args.outpath+'.hdf5', 'w')
    #outh5.create_dataset('train_data_gen', data=imgs)
    #outh5.create_dataset('train_data_raw', data=gimgs)
    outh5.create_dataset('val_data_gen', data=vimgs)
    outh5.create_dataset('val_data_raw', data=vgimgs)
    outh5.flush()
    outh5.close()


if __name__=='__main__':
    main()

