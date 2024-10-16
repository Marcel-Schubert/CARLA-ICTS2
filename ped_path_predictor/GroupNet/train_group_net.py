import sys

from matplotlib import pyplot as plt

sys.path.append(".")
sys.path.append("/workspace/data/CARLA-ICTS")

from ped_path_predictor.GroupNet.model.GroupNet_nba import GroupNet
import argparse


import logging
import os
import time
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from datetime import datetime as dt
from ped_path_predictor.new_util import getDataloadersMulti, singleDatasetsDynGroup


# path_int = "./ped_path_predictor/data/new_car/all_int.npy"
# path_non_int = "./ped_path_predictor/data/new_car/all_non_int.npy"

# path_int_car = "./ped_path_predictor/data/new_car/all_int_car.npy"
# path_non_int_car = "./ped_path_predictor/data/new_car/all_non_int_car.npy"

paths_p1 = ["./P3VI/data/01_multi_p1.npy", "./P3VI/data/02_multi_p1.npy"]
paths_p2 = ["./P3VI/data/01_multi_p2.npy", "./P3VI/data/02_multi_p2.npy"]
paths_car = ["./P3VI/data/01_multi_car.npy", "./P3VI/data/02_multi_car.npy"]


n_obs = 60
n_pred = 80
batch_size = 512
lr = 0.001

epoch_limit = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--dataset", default="nba")
parser.add_argument("--batch_size", type=int, default=batch_size)
parser.add_argument("--past_length", type=int, default=n_obs)
parser.add_argument("--future_length", type=int, default=n_pred)
parser.add_argument("--traj_scale", type=int, default=1)
parser.add_argument("--learn_prior", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=lr)
parser.add_argument("--sample_k", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--decay_step", type=int, default=10)
parser.add_argument("--decay_gamma", type=float, default=0.5)
parser.add_argument("--iternum_print", type=int, default=100)

parser.add_argument("--ztype", default="gaussian")
parser.add_argument("--zdim", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--hyper_scales", nargs="+", type=int, default=[0])
parser.add_argument("--num_decompose", type=int, default=2)
parser.add_argument("--min_clip", type=float, default=0.0)

parser.add_argument("--model_save_dir", default="saved_models/nba")
parser.add_argument("--model_save_epoch", type=int, default=5)

parser.add_argument("--epoch_continue", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--test", action="store_true", default=False)
args = parser.parse_args()


class GroupNetWrapper:

    def __init__(self, path=None):
        start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")
        obs_str = f"obs{n_obs}_pred{n_pred}"
        self.base_path = f"./_out/{self.__class__.__name__}/{obs_str}/{start_time_str}"
        os.makedirs(self.base_path, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            filename=f"{self.base_path}/train.log",
            format="%(asctime)s %(name)s: %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.logger.info(f"Model: GroupNet")
        self.logger.info(f"Observation Frames: {n_obs}")
        self.logger.info(f"Prediction Frames: {n_pred}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Initital Learning Rate: {lr}")

        self.model = GroupNet(args, "cuda:0").cuda()

        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr, eps=1e-4)
        # self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5, verbose=True)

        if path is not None:
            self.model.load_state_dict(torch.load(path))

        self.train_loader, self.test_loader, self.val_loader = getDataloadersMulti(
            paths_p1, paths_p2, paths_car, n_obs, n_pred, batch_size=batch_size, absolute=False
        )

        self.logger.info(f"Train-Batches {len(self.train_loader)}")
        self.logger.info(f"Test-Batches {len(self.test_loader)}")

    def transform(self, x, y):
        data = {}

        p1_in = x[:, :, 0:2]
        p1_cog = x[:, :, 2:4]
        p2_in = x[:, :, 4:6]
        p2_cog = x[:, :, 6:8]
        car_in = x[:, :, 8:]
        p1_out = y[:, :, 0:2]
        p2_out = y[:, :, 2:4]
        car_out = y[:, :, 4:6]

        # ped_in = x[:, :, :2].cuda()
        # cf_in = x[:, :, 2:4].cuda()
        # car_in = x[:, :, 4:].cuda()
        # ego_out = y[:, :, :2].cuda()
        # car_out = y[:, :, 2:].cuda()

        data["past_traj"] = torch.stack([p1_in, p2_in, car_in], dim=1).cuda()
        data["future_traj"] = torch.stack([p1_out, p2_out, car_out], dim=1).cuda()

        return data

    # def l2_loss_fde(self, pred, data):
    #     fde_loss = torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1)
    #     ade_loss = (
    #         torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1)
    #         .mean(dim=2)
    #         .transpose(0, 1)
    #     )
    #     loss, min_inds = (fde_loss + ade_loss).min(dim=1)
    #     return 100.0 * loss.mean()

    def _compute_joint_errors(self, data, traj_preds):
        with torch.no_grad():
            traj_gt = data["future_traj"]

            agent_num = data["past_traj"].shape[1]
            traj_preds = torch.reshape(traj_preds, (batch_size, agent_num, n_pred, 2))

            ade_losses = (torch.norm(traj_preds - traj_gt, 2, dim=-1)).cpu().numpy()
            ade_loss = np.nanmean(ade_losses, axis=(1, 2)).sum()

            fde_losses = (torch.norm(traj_preds[:, :, -1, :] - traj_gt[:, :, -1, :], 2, dim=-1)).cpu().numpy()
            fde_loss = np.nanmean(fde_losses, axis=1).sum()

        return ade_losses, fde_losses, ade_loss, fde_loss

    def train(self):

        # eval variables
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0

        for epoch in range(0, 10000):
            print(f"Epoch {epoch}")
            did_epoch_better = False

            self.model.train()

            t_before = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                print(f"\rBatch {i}/{len(self.train_loader)}", end="")

                data = self.transform(x, y)

                total_loss, loss_pred, loss_recover, loss_kl, loss_diverse = self.model(data)

                # pred: [1, 80, 512, 2]
                # data: [512, 80, 5]
                self.optimiser.zero_grad()
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch:4}, Batch: {i:4} Loss: {total_loss:.6f} Time: {(time.time() - t_before): 4.4f}"
                    )
                    t_before = time.time()

                    eval_loss, fde_loss = self.eval(self.val_loader)

                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                        did_epoch_better = True
                        self.logger.info(f"Saving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                        torch.save(self.model.state_dict(), self.base_path + f"/model_{epoch}.pth")
                    self.model.train()

            if did_epoch_better:
                self.logger.info(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 100:
                self.logger.info(f"Stopping training, no improvement in 100 epochs saved{last_best_epoch}")
                break
            # self.optimiser_scheduler.step()

    def eval(self, dataloader):
        eval_loss = 0
        fde_loss = 0
        self.model.eval()
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(dataloader):
                print(f"\rBatch {j}/{len(dataloader)}", end="")
                data = self.transform(x_val, y_val)

                diverse_pred_traj = self.model.inference(data)

                ade_losses, fde_losses, a, f = self._compute_joint_errors(data, diverse_pred_traj)

                eval_loss += a
                fde_loss += f
        self.model.train()
        eval_loss /= len(dataloader) * batch_size
        fde_loss /= len(dataloader) * batch_size
        return eval_loss, fde_loss


if __name__ == "__main__":
    if "--test" in sys.argv:

        #
        abw_eval = GroupNetWrapper(path="./ped_path_predictor/saved_models/15_20/group_net.pth")
        #
        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/01_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/01_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_1_a, int_1_f = abw_eval.eval(dl)
        print(f"\nINT-1 {int_1_a:.4f} {int_1_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/02_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/02_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_2_a, int_2_f = abw_eval.eval(dl)
        print(f"\nINT-2 {int_2_a:.4f} {int_2_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/03_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/03_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_3_a, int_3_f = abw_eval.eval(dl)
        print(f"\nINT-3 {int_3_a:.4f} {int_3_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/04_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/04_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_4_a, int_4_f = abw_eval.eval(dl)
        print(f"\nINT-1 {int_4_a:.4f} {int_4_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/05_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/05_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_5_a, int_5_f = abw_eval.eval(dl)
        print(f"\nINT-5 {int_5_a:.4f} {int_5_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/06_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/06_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_6_a, int_6_f = abw_eval.eval(dl)
        print(f"\nINT-6 {int_6_a:.4f} {int_6_f:.4f}")

        # print("\npure test", abw_eval.eval(abw_eval.test_loader))

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/01_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/01_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_1_a, non_int_1_f = abw_eval.eval(dl)
        print(f"\nNON-INT-1 {non_int_1_a:.4f} {non_int_1_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/02_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/02_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_2_a, non_int_2_f = abw_eval.eval(dl)
        print(f"\nNON-INT-2 {non_int_2_a:.4f} {non_int_2_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/03_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/03_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_3_a, non_int_3_f = abw_eval.eval(dl)
        print(f"\nNON-INT-3 {non_int_3_a:.4f} {non_int_3_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/04_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/04_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_4_a, non_int_4_f = abw_eval.eval(dl)
        print(f"\nNON-INT-4 {non_int_4_a:.4f} {non_int_4_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/05_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/05_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_5_a, non_int_5_f = abw_eval.eval(dl)
        print(f"\nNON-INT-5 {non_int_5_a:.4f} {non_int_5_f:.4f}")

        dl = singleDatasetsDynGroup(
            (
                "./ped_path_predictor/data/new_car/06_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/06_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_6_a, non_int_6_f = abw_eval.eval(dl)
        print(f"\nNON-INT-6 {non_int_6_a:.4f} {non_int_6_f:.4f}")

        print("\n\n\n")
        print("INT")
        print(
            f"&GroupNet & {int_1_a:.4f}/{int_1_f:.4f} & {int_2_a:.4f}/{int_2_f:.4f} & {int_3_a:.4f}/{int_3_f:.4f} & {int_4_a:.4f}/{int_4_f:.4f} & {int_5_a:.4f}/{int_5_f:.4f} & {int_6_a:.4f}/{int_6_f:.4f} \\\\"
        )
        print("NON-INT")
        print(
            f"&GroupNet & {non_int_1_a:.4f}/{non_int_1_f:.4f} & {non_int_2_a:.4f}/{non_int_2_f:.4f} & {non_int_3_a:.4f}/{non_int_3_f:.4f} & {non_int_4_a:.4f}/{non_int_4_f:.4f} & {non_int_5_a:.4f}/{non_int_5_f:.4f} & {non_int_6_a:.4f}/{non_int_6_f:.4f} \\\\"
        )
    else:
        abw = GroupNetWrapper()
        abw.train()
