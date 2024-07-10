import os
import random
from argparse import ArgumentParser
from math import log10, sqrt

import losses
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from utils import datasets, utils
from lpips import LPIPS
from models.UNet.model import Unet3D, Unet3D_multi
from models.VoxelMorph.model import VoxelMorph
from natsort import natsorted
from torch.utils.data import DataLoader
from models.feature_extract.model import FeatureExtract


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def evaluate(original, generated):
    max_pixel = 1.0

    #### PSNR
    mse = torch.mean((original - generated) ** 2)
    psnr = 20 * log10(max_pixel / sqrt(mse))

    ### NCC
    criterion_ncc = losses.NCC()
    ncc = criterion_ncc(original, generated)

    #### SSIM
    criterion_ssim = losses.SSIM3D()
    ssim = 1 - criterion_ssim(original, generated)

    #### NMSE
    nmse = mse / torch.mean(original**2)

    #### LPIPS
    lpips = utils.AverageMeter()
    new_original_image, new_generated_image = [], []
    for i in range(generated.shape[2]):
        new_original_image.append((original[0, 0, i, :, :] * 2 - 1).repeat(3, 1, 1))
        new_generated_image.append((generated[0, 0, i, :, :] * 2 - 1).repeat(3, 1, 1))
    for i in range(generated.shape[3]):
        new_original_image.append((original[0, 0, :, i, :] * 2 - 1).repeat(3, 1, 1))
        new_generated_image.append((generated[0, 0, :, i, :] * 2 - 1).repeat(3, 1, 1))
    original_image = torch.stack(new_original_image, dim=0).cuda()
    generated_image = torch.stack(new_generated_image, dim=0).cuda()
    lpips_ = loss_fn_alex(original_image, generated_image)
    for l in lpips_:
        lpips.update(l.item())

    new_original_image, new_generated_image = [], []
    for i in range(generated.shape[4]):
        new_original_image.append((original[0, 0, :, :, i] * 2 - 1).repeat(3, 1, 1))
        new_generated_image.append((generated[0, 0, :, :, i] * 2 - 1).repeat(3, 1, 1))
    original_image = torch.stack(new_original_image, dim=0).cuda()
    generated_image = torch.stack(new_generated_image, dim=0).cuda()
    lpips_ = loss_fn_alex(original_image, generated_image)
    for l in lpips_:
        lpips.update(l.item())

    return psnr, -1 * ncc.item(), ssim.item(), nmse.item(), lpips.avg


def main(args):
    set_seed(args.seed)

    save_dir = f"experiments/{args.dataset}"

    if args.dataset == "cardiac":
        img_size = (128, 128, 32)
        split = 90 if args.split is None else args.split
    elif args.dataset == "lung":
        img_size = (128, 128, 128)
        split = 68 if args.split is None else args.split

    """
    Initialize model
    """
    flow_model = VoxelMorph(img_size).cuda()
    if args.feature_extract:
        refinement_model = Unet3D_multi(img_size).cuda()
    else:
        refinement_model = Unet3D(img_size).cuda()

    if args.feature_extract:
        feature_model = FeatureExtract().cuda()

    best_model_path = natsorted(os.listdir(save_dir))[args.model_idx]
    best_model = torch.load(os.path.join(save_dir, best_model_path))
    print("Model: {} loaded!".format(best_model_path))
    flow_model.load_state_dict(best_model["flow_model_state_dict"])
    refinement_model.load_state_dict(best_model["model_state_dict"])
    if args.feature_extract:
        feature_model.load_state_dict(best_model["feature_model_state_dict"])

    """
    Initialize spatial transformation function
    """
    reg_model = utils.register_model(img_size, "nearest")
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, "bilinear")
    reg_model_bilin.cuda()

    """
    Initialize training
    """
    if args.dataset == "cardiac":
        data_dir = os.path.join("dataset", "ACDC", "database", "training")
        val_set = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)
    elif args.dataset == "lung":
        data_dir = os.path.join("dataset", "4D-Lung-Preprocessed")
        val_set = datasets.LungDataset(data_dir, phase="test", split=split)
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    global loss_fn_alex
    loss_fn_alex = LPIPS(net="alex").cuda()

    psnr_log = utils.AverageMeter()
    nmse_log = utils.AverageMeter()
    ncc_log = utils.AverageMeter()
    lpips_log = utils.AverageMeter()
    ssim_log = utils.AverageMeter()

    for data_idx, data in enumerate(val_loader):
        data = [t.cuda() for t in data]
        i0 = data[0]
        i1 = data[1]
        idx0, idx1 = data[2], data[3]
        video = data[4]

        i0_i1 = torch.cat((i0, i1), dim=1)
        _, _, flow_0_1, flow_1_0 = flow_model(i0_i1)

        for i in range(idx0 + 1, idx1):
            alpha = (i - idx0) / (idx1 - idx0)

            flow_0_alpha = flow_0_1 * alpha
            flow_1_alpha = flow_1_0 * (1 - alpha)

            i_0_alpha = reg_model_bilin([i0, flow_0_alpha.float()])
            i_1_alpha = reg_model_bilin([i1, flow_1_alpha.float()])

            i_alpha_combined = (1 - alpha) * i_0_alpha + alpha * i_1_alpha

            if args.feature_extract:
                x_feat_i0_list = feature_model(i0)
                x_feat_i1_list = feature_model(i1)
                x_feat_i0_alpha_list, x_feat_i1_alpha_list = [], []

                for idx in range(len(x_feat_i0_list)):
                    reg_model_feat = utils.register_model(
                        tuple([x // (2**idx) for x in img_size])
                    )
                    x_feat_i0_alpha_list.append(
                        reg_model_feat(
                            [
                                x_feat_i0_list[idx],
                                F.interpolate(
                                    flow_0_alpha * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_i1_alpha_list.append(
                        reg_model_feat(
                            [
                                x_feat_i1_list[idx],
                                F.interpolate(
                                    flow_1_alpha * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )

                i_alpha_out_diff = refinement_model(
                    i_alpha_combined, x_feat_i0_alpha_list, x_feat_i1_alpha_list
                )

            else:
                i_alpha_out_diff = refinement_model(i_alpha_combined)

            i_alpha = i_alpha_combined + i_alpha_out_diff

            gt_alpha = video[..., i]

            psnr, ncc, ssim, nmse, lpips = evaluate(gt_alpha, i_alpha)

            psnr_log.update(psnr)
            nmse_log.update(nmse)
            ncc_log.update(ncc)
            lpips_log.update(lpips)
            ssim_log.update(ssim)

    print(
        "AVG\tPSNR: {:2.2f}, NCC: {:.3f}, SSIM: {:.3f}, NMSE: {:.3f}, LPIPS: {:.3f}".format(
            psnr_log.avg,
            ncc_log.avg,
            ssim_log.avg,
            nmse_log.avg * 100,
            lpips_log.avg * 100,
        )
    )
    print(
        "STDERR\tPSNR: {:.3f}, NCC: {:.3f}, SSIM: {:.3f}, NMSE: {:.3f}, LPIPS: {:.3f}\n".format(
            psnr_log.stderr,
            ncc_log.stderr,
            ssim_log.stderr,
            nmse_log.stderr * 100,
            lpips_log.stderr * 100,
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)

    parser.add_argument(
        "--dataset", type=str, default="cardiac", choices=["cardiac", "lung"]
    )
    parser.add_argument("--model_idx", type=int, default=-1)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--feature_extract", action="store_true", default=True)

    args = parser.parse_args()

    """
    GPU configuration
    """
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))

    main(args)
