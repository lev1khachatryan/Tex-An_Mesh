import os
import cv2
import numpy as np 

from pathlib import Path
import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', type=str)
    parser.add_argument('-o', '--out_path', type=str, default='../sample_images')
    args = parser.parse_args()

    im = Image.open(open(args.input_image, 'rb'))
    R, G, B = im.convert('RGB').split()
    r = R.load()
    g = G.load()
    b = B.load()
    w, h = im.size

    # Convert non-black pixels to white
    for i in range(w):
    	for j in range(h):
    		if(r[i, j] != 0 or g[i, j] != 0 or b[i, j] != 0):
    			r[i, j] = 255 # Just change R channel

    msk_new = Image.merge('RGB', (R, R, R))
    img_name = Path(args.input_image).stem
    # cv2.imwrite(os.path.join(args.out_path, img_name + '_mask.png'), msk_new)
    msk_new.save(os.path.join(args.out_path, img_name + '_mask.png'))

if __name__ == "__main__":
    main()