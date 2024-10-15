import sys

from matplotlib import pyplot as plt

sys.path.append("/workspace/data/CARLA-ICTS")
sys.path.append(".")
import logging
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from datetime import datetime as dt
from ped_path_predictor.new_util import singleDatasets, getDataloadersMulti
from ped_path_predictor.autobots.AutoBots.models.autobot_joint import AutoBotJoint
from ped_path_predictor.autobots.AutoBots.utils.train_helpers import nll_loss_multimodes_joint


path_p1 = "./P3VI/data/01_multi_p1.npy"
path_p2 = "./P3VI/data/01_multi_p2.npy"
path_car = "./P3VI/data/01_multi_car.npy"

n_obs = 60
n_pred = 80
batch_size = 512
lr = 0.001

kl_weight = 20
entropy_weight = 40

epoch_limit = 1000


class AutoBotWrapperTwoPed:

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

        self.logger.info(f"Model: AutoBotTwoPed")
        self.logger.info(f"Observation Frames: {n_obs}")
        self.logger.info(f"Prediction Frames: {n_pred}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Initital Learning Rate: {lr}")

        self.model = AutoBotJoint(
            k_attr=4,  # default
            d_k=128,  # default
            _M=2,  # modified
            c=1,  # modified
            T=n_pred,
            L_enc=1,  # default
            dropout=0.0,  # default
            num_heads=16,  # default
            L_dec=1,  # default
            tx_hidden_size=384,  # default
            use_map_lanes=False,  # default
            num_agent_types=2,
        ).cuda()

        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr)
        # self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5, verbose=True)

        if path is not None:
            self.model.load_state_dict(torch.load(path))

        self.train_loader, self.test_loader, self.val_loader = getDataloadersMulti(
            path_p1, path_p2, path_car, n_obs, n_pred, batch_size=batch_size
        )

        self.logger.info(f"Train-Batches {len(self.train_loader)}")
        self.logger.info(f"Test-Batches {len(self.test_loader)}")

    def transform(self, x, y):
        p1_in = x[:, :, 0:2]
        p1_cog = x[:, :, 2:4]
        p2_in = x[:, :, 4:6]
        p2_cog = x[:, :, 6:8]
        car_in = x[:, :, 8:]
        car_cog = -1 * torch.ones((batch_size, n_obs, 2))
        ego_out = y[:, :, 0:2]
        ex_mask_ego = torch.ones((ego_out.shape[0], ego_out.shape[1], 1)).float()
        ego_out = torch.concatenate((ego_out, ex_mask_ego), dim=-1).cuda()

        agents_out = torch.reshape(y[:, :, 2:], (batch_size, n_pred, 2, 2))
        ex_mask_agents = torch.ones((agents_out.shape[0], agents_out.shape[1], agents_out.shape[2], 1)).float()
        agents_out = torch.concatenate((agents_out, ex_mask_agents), dim=-1).cuda()

        ego_in = torch.concatenate((p1_in, p1_cog), dim=-1)
        p2_in = torch.concatenate((p2_in, p2_cog), dim=-1)
        car_in = torch.concatenate((car_in, car_cog), dim=-1)
        agents_in = torch.concatenate((p2_in[:, :, np.newaxis, :], car_in[:, :, np.newaxis, :]), dim=2)

        map_lanes = torch.zeros((batch_size, 1, 1)).cuda()

        ped_agents = torch.ones(self.model._M + 1)
        ped_agents[-1] = 0
        car_agents = torch.zeros(self.model._M + 1)
        car_agents[-1] = 1
        agent_types = torch.stack((ped_agents, car_agents), dim=-1).repeat(batch_size, 1, 1).cuda()

        ex_mask_ego = torch.ones((ego_in.shape[0], ego_in.shape[1], 1)).float()
        ex_mask_agents = torch.ones((agents_in.shape[0], agents_in.shape[1], agents_in.shape[2], 1)).float()
        ego_in = torch.concatenate((ego_in, ex_mask_ego), dim=-1).cuda()
        agents_in = torch.concatenate((agents_in, ex_mask_agents), dim=-1).cuda()
        return ego_in, agents_in, map_lanes, ego_out, agents_out, agent_types

    def _compute_ego_errors(self, ego_preds, ego_gt, ego_in=None):
        with torch.no_grad():
            ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
            ade_losses = (
                torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1)
                .transpose(0, 1)
                .cpu()
                .numpy()
            )
            fde_losses = (
                torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
            )

            a, f = (
                torch.square(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2]).sum(-1).sqrt().sum().item(),
                torch.square((ego_preds[:, -1:, :, :2] - ego_gt[:, -1:, :, :2])).sum(-1).sqrt().sum().item(),
            )

            # some = False
            # if some == True:
            # index = 435
            # # # make output relative to the last observed frame
            # i_t = ego_in[:, 60 - 1 :, 0:2].detach().cpu().numpy()
            # i_t = np.expand_dims(i_t, axis=1)
            # i_t = np.repeat(i_t, 80, axis=1)
            # i_t = i_t.squeeze(2)

            # ego_gt = ego_gt.squeeze(0).transpose(0, 1)
            # ego_gt = ego_gt[:, :, :2].cpu().numpy() + i_t

            # ego_preds = ego_preds.squeeze(0).transpose(0, 1)
            # ego_preds = ego_preds[:, :, :2].cpu().numpy() + i_t

            # plt.plot(ego_in[index, :, 0].cpu().numpy(), ego_in[index, :, 1].cpu().numpy())
            # plt.plot(ego_gt[index, :, 0], ego_gt[index, :, 1])
            # plt.plot(ego_preds[index, :, 0], ego_preds[index, :, 1])

            # plt.xlim(70, 100)
            # plt.ylim(220, 250)

            # plt.title("Sample Trajectory Prediction (Interactive 1)\n AutoBot")
            # plt.legend(["Observed", "Ground Truth", "Predicted"])
            # plt.xlabel("X-Coordinate")
            # plt.ylabel("Y-Coordinate")
            # plt.savefig("./pics/AutoBot_int2_435.svg", dpi=300)
            # plt.savefig("./pics/AutoBot_int2_435.png", dpi=300)
            # plt.show()

        return ade_losses, fde_losses, a, f

    def _compute_joint_errors(self, preds, ego_gt, agents_gt):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        ade_losses = (
            (torch.norm(preds[0, :, :, :, :2].transpose(0, 1) - agents_gt[:, :, :, :2], 2, dim=-1)).cpu().numpy()
        )
        ade_loss = np.nanmean(ade_losses, axis=(1, 2)).sum()

        fde_losses = (torch.norm(preds[0, -1, :, :, :2] - agents_gt[:, -1, :, :2], 2, dim=-1)).cpu().numpy()
        fde_loss = np.nanmean(fde_losses, axis=1).sum()

        return ade_losses, fde_losses, ade_loss, fde_loss

    def train(self):

        # eval variables
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0

        for epoch in range(0, 1000):
            print(f"Epoch {epoch}")
            did_epoch_better = False

            self.model.train()

            t_before = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                print(f"\rBatch {i}/{len(self.train_loader)}", end="")

                ego_in, agents_in, map_lanes, ego_out, agents_out, agent_types = self.transform(x, y)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes, agent_types)

                nll_loss, kl_loss, post_entropy, ade_fde_loss = nll_loss_multimodes_joint(
                    pred_obs,
                    ego_out,
                    agents_out,
                    mode_probs,
                    entropy_weight=entropy_weight,
                    kl_weight=kl_weight,
                    use_FDEADE_aux_loss=True,
                    predict_yaw=False,
                    agent_types=agent_types,
                )

                self.optimiser.zero_grad()
                # (nll_loss + ade_fde_loss + kl_loss).backward()
                ade_fde_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch:4}, Batch: {i:4} Loss: {ade_fde_loss:.6f} Time: {(time.time() - t_before): 4.4f}"
                    )
                    t_before = time.time()

                    eval_loss, fde_loss = self.eval(self.val_loader)
                    print(nll_loss, ade_fde_loss, kl_loss)
                    print(eval_loss, fde_loss)

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
            if epoch - last_best_epoch > 10:
                self.logger.info(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                break
            # self.optimiser_scheduler.step()

    def eval(self, dataloader):
        eval_loss = 0
        fde_loss = 0
        self.model.eval()
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(dataloader):
                print(f"\rBatch {j+1}/{len(dataloader)}", end="")
                ego_in, agents_in, map_lanes, ego_out, agents_out, agent_types = self.transform(x_val, y_val)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes, agent_types)

                ade_losses, fde_losses, a, f = self._compute_joint_errors(pred_obs, ego_out, agents_out)

                eval_loss += a
                fde_loss += f
        self.model.train()
        eval_loss /= len(dataloader) * batch_size
        fde_loss /= len(dataloader) * batch_size
        return eval_loss, fde_loss


if __name__ == "__main__":
    if "--test" in sys.argv:

        abw_eval = AutoBotWrapperTwoPed(
            path="C:/Users/Marcel/Documents/MSc/Software + Data/CARLA-ICTS2/_out/AutoBotWrapperTwoPed/obs60_pred80/2024-10-14_18-31-35/model_78.pth"
        )
        print("\npure test", abw_eval.eval(abw_eval.test_loader))
        print("\npure val", abw_eval.eval(abw_eval.val_loader))
        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/01_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/01_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # int_1_a, int_1_f = abw_eval.eval(dl)
        # print(f"\nINT-1 {int_1_a:.4f} {int_1_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/02_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/02_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # int_2_a, int_2_f = abw_eval.eval(dl)
        # print(f"\nINT-2 {int_2_a:.4f} {int_2_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/03_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/03_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # int_3_a, int_3_f = abw_eval.eval(dl)
        # print(f"\nINT-3 {int_3_a:.4f} {int_3_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/04_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/04_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # int_4_a, int_4_f = abw_eval.eval(dl)
        # print(f"\nINT-1 {int_4_a:.4f} {int_4_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/05_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/05_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # int_5_a, int_5_f = abw_eval.eval(dl)
        # print(f"\nINT-5 {int_5_a:.4f} {int_5_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/06_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/06_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # int_6_a, int_6_f = abw_eval.eval(dl)
        # print(f"\nINT-6 {int_6_a:.4f} {int_6_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/01_non_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/01_non_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # non_int_1_a, non_int_1_f = abw_eval.eval(dl)
        # print(f"\nNON-INT-1 {non_int_1_a:.4f} {non_int_1_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/02_non_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/02_non_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # non_int_2_a, non_int_2_f = abw_eval.eval(dl)
        # print(f"\nNON-INT-2 {non_int_2_a:.4f} {non_int_2_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/03_non_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/03_non_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # non_int_3_a, non_int_3_f = abw_eval.eval(dl)
        # print(f"\nNON-INT-3 {non_int_3_a:.4f} {non_int_3_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/04_non_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/04_non_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # non_int_4_a, non_int_4_f = abw_eval.eval(dl)
        # print(f"\nNON-INT-4 {non_int_4_a:.4f} {non_int_4_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/05_non_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/05_non_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # non_int_5_a, non_int_5_f = abw_eval.eval(dl)
        # print(f"\nNON-INT-5 {non_int_5_a:.4f} {non_int_5_f:.4f}")

        # dl = singleDatasets(
        #     (
        #         "./ped_path_predictor/data/new_car/06_non_int_cleaned.npy",
        #         "./ped_path_predictor/data/new_car/06_non_int_cleaned_car.npy",
        #     ),
        #     n_obs,
        #     n_pred,
        #     512,
        # )
        # non_int_6_a, non_int_6_f = abw_eval.eval(dl)
        # print(f"\nNON-INT-6 {non_int_6_a:.4f} {non_int_6_f:.4f}")

        # print("\n\n\n")
        # print("INT")
        # print(
        #     f"&AutoBot & {int_1_a:.4f}/{int_1_f:.4f} & {int_2_a:.4f}/{int_2_f:.4f} & {int_3_a:.4f}/{int_3_f:.4f} & {int_4_a:.4f}/{int_4_f:.4f} & {int_5_a:.4f}/{int_5_f:.4f} & {int_6_a:.4f}/{int_6_f:.4f} \\\\"
        # )
        # print("NON-INT")
        # print(
        #     f"&AutoBot & {non_int_1_a:.4f}/{non_int_1_f:.4f} & {non_int_2_a:.4f}/{non_int_2_f:.4f} & {non_int_3_a:.4f}/{non_int_3_f:.4f} & {non_int_4_a:.4f}/{non_int_4_f:.4f} & {non_int_5_a:.4f}/{non_int_5_f:.4f} & {non_int_6_a:.4f}/{non_int_6_f:.4f} \\\\"
        # )

    else:
        abw = AutoBotWrapperTwoPed()
        abw.train()
