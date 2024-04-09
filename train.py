import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

import losses
from utils import datasets, utils
from models.UNet.model import Unet3D, Unet3D_multi
from models.VoxelMorph.model import VoxelMorph
from models.feature_extract.model import FeatureExtract


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def main(args):
    set_seed(args.seed)

    if not os.path.exists(f"experiments/{args.dataset}"):
        os.makedirs(f"experiments/{args.dataset}")

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

    """
    Initialize spatial transformation function
    """
    reg_model = utils.register_model(img_size, "nearest").cuda()
    reg_model_bilin = utils.register_model(img_size, "bilinear").cuda()
    for param in reg_model.parameters():
        param.requires_grad = False
    for param in reg_model_bilin.parameters():
        param.requires_grad = False

    """
    Initialize training
    """
    if args.dataset == "cardiac":
        data_dir = os.path.join("dataset", "ACDC", "database", "training")
        train_set = datasets.ACDCHeartDataset(data_dir, phase="train", split=split)
        val_set = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)
    elif args.dataset == "lung":
        data_dir = os.path.join("dataset", "4D-Lung-Preprocessed")
        train_set = datasets.LungDataset(data_dir, phase="train", split=split)
        val_set = datasets.LungDataset(data_dir, phase="test", split=split)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    if args.feature_extract:
        optimizer = optim.Adam(
            list(flow_model.parameters())
            + list(refinement_model.parameters())
            + list(feature_model.parameters()),
            lr=args.lr,
            weight_decay=0,
            amsgrad=True,
        )
    else:
        optimizer = optim.Adam(
            list(flow_model.parameters()) + list(refinement_model.parameters()),
            lr=args.lr,
            weight_decay=0,
            amsgrad=True,
        )
    criterion_ncc = losses.NCC()
    criterion_cha = losses.CharbonnierLoss
    criterion_reg = losses.Grad3d(penalty="l2")
    criterion_l1n = losses.L1_norm()
    epsilon = 1e-3

    best_ncc = -1
    wandb.init(project="UVI-Net", name=args.dataset, config=args)

    for epoch in trange(args.max_epoch):
        """
        Training
        """
        loss_all = utils.AverageMeter()
        loss_all_full = utils.AverageMeter()
        loss_ncc_all_full = utils.AverageMeter()
        loss_cha_all_full = utils.AverageMeter()
        loss_reg_all_full = utils.AverageMeter()
        loss_all_cycle = utils.AverageMeter()
        loss_diff_all = utils.AverageMeter()

        for idx, data in enumerate(train_loader):
            refinement_model.train()
            flow_model.train()

            data = [t.cuda() for t in data]
            i0 = data[0]
            i1 = data[1]

            alpha1 = random.uniform(-0.5, 0.0)
            alpha2 = random.uniform(0.0, 1.0)
            alpha3 = random.uniform(1.0, 1.5)

            i0_i1 = torch.cat((i0, i1), dim=1)
            i_0_1, i_1_0, flow_0_1, flow_1_0 = flow_model(i0_i1)

            loss_ncc_1 = criterion_ncc(i_0_1, i1) * args.weight_ncc
            loss_cha_1 = criterion_cha(i_0_1, i1, eps=epsilon) * args.weight_cha
            loss_reg_1 = criterion_reg(flow_0_1, None)
            loss_ncc_0 = criterion_ncc(i_1_0, i0) * args.weight_ncc
            loss_cha_0 = criterion_cha(i_1_0, i0, eps=epsilon) * args.weight_cha
            loss_reg_0 = criterion_reg(flow_1_0, None)
            loss_full = (
                loss_ncc_1 + loss_cha_1 + loss_reg_1 + \
                loss_ncc_0 + loss_cha_0 + loss_reg_0
            )

            loss_all_full.update(loss_full.item(), i1.numel())
            loss_ncc_all_full.update(loss_ncc_1.item(), i1.numel())
            loss_cha_all_full.update(loss_cha_1.item(), i1.numel())
            loss_reg_all_full.update(loss_reg_1.item(), i1.numel())
            loss_ncc_all_full.update(loss_ncc_0.item(), i1.numel())
            loss_cha_all_full.update(loss_cha_0.item(), i1.numel())
            loss_reg_all_full.update(loss_reg_0.item(), i1.numel())

            if args.weight_cycle == 0:
                optimizer.zero_grad()
                loss_full.backward()
                optimizer.step()

                loss_all.update(loss_full.item(), i0.numel())

                continue

            """
            First Interpolation
            """
            flow_0_a1 = flow_0_1 * alpha1
            i_0_a1 = reg_model_bilin([i0, flow_0_a1.float()])

            if alpha2 < 0.5:
                flow_0_a2 = flow_0_1 * alpha2
                i_0_a2 = reg_model_bilin([i0, flow_0_a2.float()])
                i_unknown_a2 = i_0_a2
            else:
                flow_1_a2 = flow_1_0 * (1 - alpha2)
                i_1_a2 = reg_model_bilin([i1, flow_1_a2.float()])
                i_unknown_a2 = i_1_a2

            flow_1_a3 = flow_1_0 * (1 - alpha3)
            i_1_a3 = reg_model_bilin([i1, flow_1_a3.float()])

            """
            Second Interpolation
            """
            ia1_ia2 = torch.cat((i_0_a1, i_unknown_a2), dim=1)
            ia2_ia3 = torch.cat((i_unknown_a2, i_1_a3), dim=1)

            _, _, flow_a1_a2, flow_a2_a1 = flow_model(ia1_ia2)
            _, _, flow_a2_a3, flow_a3_a2 = flow_model(ia2_ia3)

            alpha12 = (0 - alpha1) / (alpha2 - alpha1)
            alpha23 = (1 - alpha2) / (alpha3 - alpha2)

            flow_a1_0 = flow_a1_a2 * alpha12
            flow_a2_0 = flow_a2_a1 * (1 - alpha12)
            flow_a2_1 = flow_a2_a3 * alpha23
            flow_a3_1 = flow_a3_a2 * (1 - alpha23)

            i_a1_0 = reg_model_bilin([i_0_a1, flow_a1_0.float()])
            i_a2_0 = reg_model_bilin([i_unknown_a2, flow_a2_0.float()])
            i_a2_1 = reg_model_bilin([i_unknown_a2, flow_a2_1.float()])
            i_a3_1 = reg_model_bilin([i_1_a3, flow_a3_1.float()])

            i0_combined = (1 - alpha12) * i_a1_0 + alpha12 * i_a2_0
            i1_combined = (1 - alpha23) * i_a2_1 + alpha23 * i_a3_1

            if args.feature_extract:
                x_feat_a1_list = feature_model(i_0_a1)
                x_feat_a2_list = feature_model(i_unknown_a2)
                x_feat_a3_list = feature_model(i_1_a3)
                (
                    x_feat_a1_0_list,
                    x_feat_a2_0_list,
                    x_feat_a2_1_list,
                    x_feat_a3_1_list,
                ) = ([], [], [], [])

                for idx in range(len(x_feat_a1_list)):
                    reg_model_feat = utils.register_model(
                        tuple([x // (2**idx) for x in img_size])
                    )
                    x_feat_a1_0_list.append(
                        reg_model_feat(
                            [
                                x_feat_a1_list[idx],
                                F.interpolate(
                                    flow_a1_0 * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_a2_0_list.append(
                        reg_model_feat(
                            [
                                x_feat_a2_list[idx],
                                F.interpolate(
                                    flow_a2_0 * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_a2_1_list.append(
                        reg_model_feat(
                            [
                                x_feat_a2_list[idx],
                                F.interpolate(
                                    flow_a2_1 * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_a3_1_list.append(
                        reg_model_feat(
                            [
                                x_feat_a3_list[idx],
                                F.interpolate(
                                    flow_a3_1 * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )

                i0_out_diff = refinement_model(
                    i0_combined, x_feat_a1_0_list, x_feat_a2_0_list
                )
                i1_out_diff = refinement_model(
                    i1_combined, x_feat_a2_1_list, x_feat_a3_1_list
                )
                
            else:
                i0_out_diff = refinement_model(i0_combined)
                i1_out_diff = refinement_model(i1_combined)

            i0_out = i0_combined + i0_out_diff
            i1_out = i1_combined + i1_out_diff
            loss_diff_0 = criterion_l1n(i0_out_diff)
            loss_diff_1 = criterion_l1n(i1_out_diff)
            loss_diff = (loss_diff_0 + loss_diff_1) * args.weight_diff

            loss_cyc_ncc_0 = criterion_ncc(i0_out, i0) * args.weight_ncc
            loss_cyc_cha_0 = criterion_cha(i0_out, i0, eps=epsilon) * args.weight_cha
            loss_cyc_ncc_1 = criterion_ncc(i1_out, i1) * args.weight_ncc
            loss_cyc_cha_1 = criterion_cha(i1_out, i1, eps=epsilon) * args.weight_cha

            loss_cycle_0 = loss_cyc_ncc_0 + loss_cyc_cha_0
            loss_cycle_1 = loss_cyc_ncc_1 + loss_cyc_cha_1
            loss_cycle = (loss_cycle_0 + loss_cycle_1) * args.weight_cycle

            loss_diff_all.update(loss_diff_0.item(), i1.numel())
            loss_diff_all.update(loss_diff_1.item(), i1.numel())
            loss_all_cycle.update(loss_cycle_0.item(), i1.numel())
            loss_all_cycle.update(loss_cycle_1.item(), i1.numel())

            loss = loss_full + loss_cycle + loss_diff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), i0.numel())

        wandb.log({"Loss_all/train": loss_all.avg}, step=epoch)
        wandb.log({"Loss_full/train_all": loss_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_img_ncc": loss_ncc_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_img_cha": loss_cha_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_reg": loss_reg_all_full.avg}, step=epoch)
        wandb.log({"Loss_cycle/train_all": loss_all_cycle.avg}, step=epoch)
        wandb.log({"Loss_cycle/train_diff": loss_diff_all.avg}, step=epoch)

        """
        Validation
        """
        if (epoch == 0) or ((epoch + 1) % 50 == 0):
            eval_ncc = utils.AverageMeter()
            with torch.no_grad():
                for data in val_loader:
                    flow_model.eval()
                    refinement_model.eval()
                    data = [t.cuda() for t in data]
                    i0 = data[0]
                    i1 = data[1]

                    i0_i1 = torch.cat((i0, i1), dim=1)

                    _, _, flow_0_1, _ = flow_model(i0_i1)

                    i_0_1 = reg_model_bilin([i0, flow_0_1.float()])

                    ncc = -1 * criterion_ncc(i_0_1, i1)
                    eval_ncc.update(ncc.item(), i0.size(0))

            print("Epoch {}, NCC {:.5f}\n".format(epoch, eval_ncc.avg), flush=True)

            if eval_ncc.avg > best_ncc:
                best_ncc = eval_ncc.avg

            torch.save(
                {
                    "epoch": epoch + 1,
                    "flow_model_state_dict": flow_model.state_dict(),
                    "model_state_dict": refinement_model.state_dict(),
                    "feature_model_state_dict": feature_model.state_dict() if args.feature_extract else None,
                    "best_ncc": best_ncc,
                    "optimizer": optimizer.state_dict(),
                },
                "experiments/{}/epoch{}_ncc{:.4f}.ckpt".format(args.dataset, epoch + 1, eval_ncc.avg),
            )

            wandb.log({"Validate/NCC": eval_ncc.avg}, step=epoch)

        loss_all.reset()
        loss_all_full.reset()
        loss_ncc_all_full.reset()
        loss_cha_all_full.reset()
        loss_reg_all_full.reset()
        loss_all_cycle.reset()
        loss_diff_all.reset()

    print("best_ncc {}".format(best_ncc), flush=True)

    wandb.finish()


def save_checkpoint(state, save_dir="models", filename="checkpoint.pth.tar"):
    torch.save(state, save_dir + filename)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="cardiac", choices=["cardiac", "lung"]
    )

    parser.add_argument("--weight_cycle", type=float, default=1.0)
    parser.add_argument("--weight_diff", type=float, default=1.0)

    parser.add_argument("--weight_ncc", type=float, default=1.0)
    parser.add_argument("--weight_cha", type=float, default=1.0)
    parser.add_argument("--feature_extract", action="store_true", default=True)
    parser.add_argument("--name", type=str, default="")

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
