import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.io as skio
import torch
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

from utils import get_files_pattern, mkdir_p, psnr, rmse, return_FPN

from utils_CP import fwd_grad, proj_J, K, trans_K
from utils_CP import proj_F_original as proj_F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default=None, type=str,
                        help='Input directory')
    parser.add_argument('--refdir', default='', type=str,
                        help='Reference directory')
    parser.add_argument('--outdir',
                        default='',
                        type=str,
                        help='Output directory')
    parser.add_argument('--pattern', default='*png',
                        help='File pattern')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed')
    ### adding noise
    parser.add_argument('--add_noise', action='store_false',
                        help='If True (=not calling --add_noise), will add noise to the image')
    parser.add_argument('--noise_type', default='interpolation', choices=['fixed', 'interpolation', 'brownian'],
                        help='Type of FPN')

    parser.add_argument('--noise_offset_level', default=5, type=int,
                        help='Level of noise')
    parser.add_argument('--noise_offset_type', default="mixed", choices=["None", "structured", "unstructured", "mixed"],
                        help='type of noise to add')

    parser.add_argument('--noise_gain_level', default=.10, type=int,
                        help='Level of noise')
    parser.add_argument('--noise_gain_type', default="None", choices=["None", "structured", "unstructured", "mixed"],
                        help='type of noise to add')

    parser.add_argument('--clip', type=bool, default=False,
                        help='If True, will clip the data')

    parser.add_argument('--data_range', default=2 ** 8 - 1, type=int,
                        help='range of the data')

    ### parameters
    parser.add_argument('--max_iter', default=1, type=int,
                        help='maximum of number of iterations')
    parser.add_argument('--lambda_O', default=5 * 10 ** -2, type=float,
                        help='Strength of the norm term in the loss')
    parser.add_argument('--nb_frames', default=16, type=int,
                        help='Number of frames used per iterations')

    ### parameters optimization
    parser.add_argument('--tau', default=2 * 10 ** -2, type=float,
                        help='Learning rate')
    parser.add_argument('--sigma', default=5 * 10 ** -1, type=float,
                        help='Learning rate')
    ### denoised

    args = parser.parse_args()
    print(args)

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.add_noise and args.indir:
        raise RuntimeError("Either you add noise to clean data, or you have noisy data")

    ## Creates output directory if it does not exist
    outdir = args.outdir
    mkdir_p(outdir)

    outdir_noisy = os.path.join(outdir, 'noisy')
    outdir_denoised = os.path.join(outdir, 'denoised')

    outdir_noisy = os.path.join(args.outdir, 'noisy')
    outdir_denoised = os.path.join(args.outdir, 'denoised')

    mkdir_p(outdir_noisy)
    mkdir_p(outdir_denoised)

    ## Write args in a text file
    #save_arg(args)

    folders = sorted(os.listdir(args.refdir))
    files_folder_0 = get_files_pattern(os.path.join(args.refdir, folders[0]), args.pattern)
    nb_files = len(files_folder_0)

    for folder_name in folders:
        outdir_folder = os.path.join(outdir_noisy, folder_name)
        mkdir_p(outdir_folder)
        outdir_folder = os.path.join(outdir_denoised, folder_name)
        mkdir_p(outdir_folder)

    img = skio.imread(os.path.join(args.refdir, os.path.join(folders[0], files_folder_0[0])))
    img_shape = img.shape
    if len(img_shape) > 2:
        H, W, C = img_shape
    elif len(img_shape) == 2:
        H, W = img_shape
        C = 1

    is_tiff = True if files_folder_0[0][-4:] == 'tiff' else False
    file_format = '.tiff' if is_tiff else files_folder_0[0][-4:]

    data_range = args.data_range
    data_type = img.dtype
    noise_offset_level = args.noise_offset_level
    noise_gain_level = args.noise_gain_level
    noise_offset_type = args.noise_offset_type
    noise_gain_type = args.noise_gain_type

    max_iter = args.max_iter

    if args.add_noise:
        print("Adding noise")
        noise_offset, noise_gain = return_FPN(noise_offset_type, noise_gain_type, noise_offset_level,
                                              noise_gain_level, H, W, C)
        if args.noise_type == 'interpolation':
            noise_offset_1, noise_gain_1 = return_FPN(noise_offset_type, noise_gain_type, noise_offset_level,
                                                      noise_gain_level, H, W, C)
            noise_offset_2, noise_gain_2 = return_FPN(noise_offset_type, noise_gain_type, noise_offset_level,
                                                      noise_gain_level, H, W, C)
    else:
        noise_gain = np.ones((H, W, C))
        noise_offset = np.zeros((H, W, C))

    nb_clips = nb_files

    O = np.zeros((H, W, C))
    N = args.nb_frames
    L_2 = 8
    tau = args.tau #1 / np.sqrt(L_2)
    sigma = args.sigma #1 / (L_2 * tau)
    gamma = 0.7 * args.lambda_O

    lambda_O = args.lambda_O
    O = np.squeeze(O)

    noise_gain = np.squeeze(noise_gain)
    noise_offset = np.squeeze(noise_offset)

    O_bar = np.copy(O)
    V = K(O, N)

    update_O_hist = []
    pnsr_hist_denoised = []
    pnsr_hist_noisy = []

    for idx_iter in tqdm(range(nb_clips)):
        images = []
        current_files = [os.path.join(args.refdir, os.path.join(folders[i], files_folder_0[idx_iter])) for i in range(N)]
        for f in current_files:
            img = skio.imread(f)
            images.append(img)
        stack_images = np.array(images)

        if args.add_noise:
            if args.noise_type == 'interpolation':
                a_1 = (1 - idx_iter / nb_clips) / np.sqrt((1 - idx_iter / nb_clips) ** 2 + (idx_iter / nb_clips) ** 2)
                a_2 = (idx_iter / nb_clips) / np.sqrt((1 - idx_iter / nb_clips) ** 2 + (idx_iter / nb_clips) ** 2)
                noise_gain = a_1 * (noise_gain_1 - 1) + a_2 * (noise_gain_2 - 1) + 1
                noise_offset = a_1 * noise_offset_1 + a_2 * noise_offset_2
            elif args.noise_type == 'brownian':
                noise_offset_b, noise_gain_b = return_FPN(noise_offset_type, noise_gain_type,
                                                          noise_offset_level / 10, noise_gain_level / 10, H, W, C)
                noise_gain = np.squeeze(noise_gain) * np.squeeze(noise_gain_b)
                noise_offset = np.squeeze(noise_offset) + np.squeeze(noise_offset_b)
            noise_gain = np.squeeze(noise_gain)
            noise_offset = np.squeeze(noise_offset)
            stack_noisy = stack_images * noise_gain + noise_offset
            if args.clip:
                stack_noisy[stack_noisy < 0] = 0
                stack_noisy[stack_noisy > data_range] = data_range
        elif args.indir:
            noisy_images = []
            for f in current_files:
                noisy_img = skio.imread(os.path.join(args.indir, f))
                noisy_images.append(noisy_img)
            stack_noisy = np.array(noisy_images)
        else:
            stack_noisy = stack_images

        Y = stack_noisy

        pnsr_hist_noisy.append(psnr(stack_noisy, stack_images, max_value=data_range))

        M_y = []
        for i in range(args.nb_frames):
            M_y.append(fwd_grad(Y[i]))

        M_y = np.array(M_y)

        for iterations in range(max_iter):

            V_next = proj_F(V + sigma * K(O_bar, N), M_y, sigma)
            O_next = proj_J(O - tau * trans_K(V_next), lambda_O, tau)

            theta = 1 / np.sqrt(1 + 2 * gamma * tau)
            tau_next = theta * tau
            sigma_next = sigma / theta

            O_bar_next = O_next + theta * (O_next - O)

            O = np.copy(O_next)
            O_bar = np.copy(O_bar_next)
            V = np.copy(V_next)
            tau = tau_next
            sigma = sigma_next

        update_O_hist.append(rmse(noise_offset, O))

        stack_denoised = Y - O
        stack_denoised[stack_denoised < 0] = 0
        stack_denoised[stack_denoised > data_range] = data_range
        stack_denoised = stack_denoised.round()

        pnsr_hist_denoised.append(psnr(stack_images, stack_denoised, max_value=data_range))

        for i in range(args.nb_frames):
            denoised_img = stack_denoised[i]
            denoised_img[denoised_img > data_range] = data_range
            denoised_img[denoised_img < 0] = 0

            # save denoised image
            denoised_img = denoised_img.round()
            denoised_img = denoised_img.astype(data_type)
            skio.imsave(os.path.join(outdir_denoised, os.path.join(folders[i], files_folder_0[idx_iter])), denoised_img)

            if args.add_noise:
                noisy_img = stack_noisy[i]
                noisy_img[noisy_img > data_range] = data_range
                noisy_img[noisy_img < 0] = 0

                # save denoised image
                noisy_img = noisy_img.round()
                noisy_img = noisy_img.astype(data_type)
                skio.imsave(os.path.join(outdir_noisy, os.path.join(folders[i], files_folder_0[idx_iter])), noisy_img)
                skio.imsave(os.path.join(outdir_noisy, os.path.join(folders[i], files_folder_0[idx_iter])), noisy_img)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 8))
    plt.xlabel("Number of iterations")
    plt.ylabel("PNSR")
    plt.plot(pnsr_hist_denoised)
    plt.savefig(os.path.join(outdir, 'psnr_hist.png'))
    plt.show()
    plt.clf()

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 8))
    plt.xlabel("Number of iterations")
    plt.ylabel("PNSR")
    plt.plot(pnsr_hist_noisy)
    plt.savefig(os.path.join(outdir, 'psnr_hist_noisy.png'))
    plt.show()
    plt.clf()

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 8))
    # plt.yscale('log')
    plt.xlabel("Number of iterations")
    plt.ylabel("Update_O")
    plt.plot(update_O_hist)
    plt.savefig(os.path.join(outdir, 'O_hist.png'))
    plt.show()
    plt.clf()

    skio.imsave(os.path.join(outdir, 'O.tiff'), O)

    skio.imsave(os.path.join(outdir, 'O_true.tiff'), noise_offset)

    print(pnsr_hist_denoised[0], pnsr_hist_denoised[-1])
    print(update_O_hist[-1])

    np.savetxt(os.path.join(outdir, 'psnr.txt'), pnsr_hist_denoised)
    np.savetxt(os.path.join(outdir, 'psnr_noisy.txt'), pnsr_hist_noisy)
    np.savetxt(os.path.join(outdir, 'update_O.txt'), update_O_hist)
