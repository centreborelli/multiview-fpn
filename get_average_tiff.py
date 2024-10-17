import os
import argparse
import errno
import fnmatch

import numpy as np
from tqdm import tqdm

import skimage.io as skio

import tifffile


def get_files_pattern(d, pattern):
    """
    List elements in the directory d with pattern.
    Sort the elements.
    """
    files = os.listdir(d)
    files = fnmatch.filter(files, pattern)
    return sorted(files)


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    if path:
        try:
            os.makedirs(path)
        except OSError as exc: # requires Python > 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: assert(False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='',
                        help=('Input directory'))
    parser.add_argument('--outdir', default='',
                        help=('Output file'))
    parser.add_argument('--pattern', default='*.tiff',
                        help=('File pattern'))
    args = parser.parse_args()
    
    folders = get_files_pattern(args.indir, '*')
    outdir = args.outdir
    mkdir_p(outdir)
    
    for folder in folders:
        folder_path = os.path.join(args.indir, folder)

        files = get_files_pattern(folder_path, args.pattern)
        dtype = None
        
        
        average_img = np.float64(np.zeros_like(np.squeeze(tifffile.imread(os.path.join(folder_path, files[0])))))
        for f in tqdm(files):
            image = np.squeeze(tifffile.imread(os.path.join(folder_path, f)))
            dtype=image.dtype
            average_img += image

        average_img = (average_img / len(files)).round()
        skio.imsave(os.path.join(outdir, folder + '.tiff'), np.asarray(average_img, dtype=dtype))
