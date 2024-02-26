import fnmatch
import os
import errno
import re
import numpy as np
import torch

import tifffile

from scipy import signal


def extract_num(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]


def get_files_pattern(d, pattern):
    """
    List elements in the directory d with pattern.
    Sort the elements.
    """
    files = os.listdir(d)
    files = fnmatch.filter(files, pattern)
    files.sort(key=extract_num)
    return files


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    if path:
        try:
            os.makedirs(path)
        except OSError as exc:  # requires Python > 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                assert False
                

def save_arg(args):
    path_arg = args.outdir + "args.txt" if args.outdir[-1] == '/' else args.outdir + '/' + "args.txt"
    write_args = open(path_arg, 'w')

    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
        write_args.write('{}: {}'.format(p, v) + '\n')
    print('\n')
    write_args.close()

    print("args saved in", path_arg)
    

def psnr(img1, img2, max_value=1):
    mse = ((img1 - img2) ** 2).mean()
    return 10 * np.log10(max_value ** 2 / mse)


def psnr_torch(img1, img2, max_value=1):
    mse = ((img1 - img2) ** 2).mean()
    return 10 * torch.log10(max_value ** 2 / mse)


def l1_norm(x):
    return (torch.abs(x)).sum()

def l2_square_norm(x):
    return (x**2).sum()


def rmse(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return np.sqrt(mse)


def rmse_torch(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return torch.sqrt(mse)


def roughness_index(img):
    h1 = np.array([[1, -1]])
    h2 = h1.T
    img = np.squeeze(img)
    if len(img.shape) == 2:
        res = l1_norm(signal.fftconvolve(img, h1, mode='full'))
        res += l1_norm(signal.fftconvolve(h2, img, mode='full'))
        res = res / l1_norm(img)
        return res
    elif len(img.shape) == 3:
        res = 0
        for i in range(3):
            res_ = l1_norm(signal.fftconvolve(img[i, :, :], h1, mode='full'))
            res_ += l1_norm(signal.fftconvolve(h2, img[i, :, :], mode='full'))
            res_ = res_ / l1_norm(img)
            res += res_ / 3
        return res
    
    
def return_gaussian_noise(mean, noise_level, H, W, C=1):
    """
    :param C: number of channels of the image we want to add noise
    :param W: width of the image we want to add noise
    :param H: height of the image we want to add noise
    :param mean: mean of the noise we want to add
    :param noise_level: standard deviation of the noise we want to add
    :return:
    """
    noise = np.random.normal(size=(H, W, C), loc=mean, scale=noise_level)
    return noise


def return_colum_row_noise(mean, noise_level, H, W, C=1, add=True):
    """
    :param C: number of channels of the image we want to add noise
    :param W: width of the image we want to add noise
    :param H: height of the image we want to add noise
    :param mean: mean of the noise we want to add
    :param noise_level: standard deviation of the noise we want to add
    :return:
    """
    noise = np.zeros((H, W, C))
    for i in range(C):
        noise_column = np.random.normal(size=(H, 1), loc=mean, scale=noise_level) @ np.ones((1, W))
        noise_row = np.ones((H, 1)) @ np.random.normal(size=(1, W), loc=mean, scale=noise_level)
        if add:
            noise[:, :, i] = noise_row + noise_column
        else:
            noise[:, :, i] = noise_row * noise_column
    return noise


def return_FPN(noise_offset_type, noise_gain_type, noise_offset_level, noise_gain_level,  H, W, C=1):
    noise_offset = np.zeros((1, H, W, C))
    noise_gain = np.ones((1, H, W, C))

    unstructured_noise_offset = return_gaussian_noise(0, noise_offset_level, H=H, W=W, C=C)
    structured_noise_offset = return_colum_row_noise(0, noise_offset_level, H=H, W=W, C=C)

    unstructured_noise_gain = return_gaussian_noise(1, noise_gain_level, H=H, W=W, C=C)
    structured_noise_gain = return_colum_row_noise(1, noise_gain_level, H=H, W=W, C=C, add=False)

    if noise_offset_type == "structured" or noise_offset_type == "mixed":
        noise_offset += structured_noise_offset
    if noise_offset_type == "unstructured" or noise_offset_type == "mixed":
        noise_offset += unstructured_noise_offset
    if noise_gain_type == "structured" or noise_gain_type == "mixed":
        noise_gain *= structured_noise_gain
    if noise_gain_type == "unstructured" or noise_gain_type == "mixed":
        noise_gain *= unstructured_noise_gain
    return noise_offset, noise_gain
