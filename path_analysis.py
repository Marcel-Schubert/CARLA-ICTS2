import numpy as np
import torch
from matplotlib import pyplot as plt

# from CI3PP.model import CI3PP
from P3VI.utils import load_data
# from CI3PP.train_60_80 import CI3PPWrapper as CI3PPWrapper_60_80
# from ped_path_predictor.m2p3_60_80 import PathPredictor
from ped_path_predictor.model import M2P3

data = "./P3VI/data/single/01_int.npy"
data2 = './P3VI/data/car_dump/01_int.npy'


n_obs = 60
n_pred = 80
sample_pos = 1920 * 74
stepsize = 1920
sample_range = 10

def plot_path(data, pred=None, actual=None):
    data = data.cpu().detach().numpy()

    plt.plot(data[:, 0], data[:, 1], 'r')

    if pred is not None:
        pred = pred.squeeze()
        pred = pred.cpu().detach().numpy()


        absolute = np.array([data[-1] + pred[0]])
        for i in range(1, pred.shape[0]):
            absolute = np.vstack((absolute, [pred[i] + absolute[i - 1]]))

        plt.plot(absolute[:, 0], absolute[:, 1], 'b', linestyle='dashed')
    #
    if actual is not None:
        actual = actual.cpu().detach().numpy()
        absolute_actual = np.array([data[-1] + actual[0]])
        for i in range(1, pred.shape[0]):
            absolute_actual = np.vstack((absolute_actual, [actual[i] + absolute_actual[i - 1]]))

        plt.plot(absolute_actual[:, 0], absolute_actual[:, 1], 'g', linestyle='dotted')


    plt.show()
    # for i in range(0, 10):
    #     plt.plot(data[i, :, 0], data[i, :, 1], 'r')
    #     # plt.plot(pred[i, :, 0], pred[i, :, 1], 'b')
    #     plt.show()

def plot_all(data):
    data = data.cpu().detach().numpy()
    for x in range(sample_range):
        plt.plot(data[:, sample_pos+x*stepsize, 0], data[:, sample_pos+x*stepsize, 1], 'r')
        print(data[:, sample_pos+x*stepsize, 2:])
    plt.show()

    # plt.plot(data[:, 255, 0], data[:, 255, 1])
    # plt.show()
    # for i in range(0, 10):
    #     plt.plot(data[i, :, 0], data[i, :, 1], 'r')
    #     plt.show()


def plot_rel(actual, pred):
    actual = actual.cpu().detach().numpy()
    pred = pred.squeeze()
    pred = pred.cpu().detach().numpy()
    plt.plot(actual[:, 0], actual[:, 1], 'r')
    plt.plot(pred[:, 0], pred[:, 1], 'b')
    plt.show()




if __name__ == '__main__':
    # obs_train, pred_train = load_data(data, 500, 0)

    # convert to np array and float32
    # input_train = np.array(obs_train[:, :, :], dtype=np.float32)
    # output_train = np.array(pred_train[:, :, :], dtype=np.float32)

    def load_full(data, n_observed_frames, n_predict_frames):
        with open(data,'rb') as f:
            raw = np.load(f, allow_pickle=True)


        enum_conv = lambda t: t.value
        vfunc = np.vectorize(enum_conv)
        raw[:,:,2] = vfunc(raw[:,:,2])
        raw[:,:,3] = vfunc(raw[:,:,3])
        raw = raw.astype(np.float32)
        window = raw.shape[1] - n_observed_frames - n_predict_frames

        observed_data, predict_data = [], []
        for k in range(0, window, 2):
            observed = raw[:, k:n_observed_frames + k, :]
            pred = raw[:, k + n_observed_frames:n_predict_frames + n_observed_frames + k, 0:2]

            observed_data.append(observed)
            predict_data.append(pred)

        observed_data = np.concatenate(observed_data, axis=0)
        predict_data = np.concatenate(predict_data, axis=0)

        return observed_data, predict_data


    obs_1, pred_1 = load_full(data, 488, 2)
    obs_2, pred_2 = load_full(data2, 488, 2)

    for i in range(1, 10):
        to_plot = obs_1[i]
        # to_plot2 = obs_2[i]
        plt.plot(to_plot[:, 0], to_plot[:, 1])
        to_plot = obs_2[i]
        # to_plot2 = obs_2[i]
        plt.plot(to_plot[:, 0], to_plot[:, 1])
        # plt.plot(to_plot2[:, 0], to_plot2[:, 1])
    plt.show()

    # for i in range(1, 10):
    #
    #     # plt.plot(to_plot2[:, 0], to_plot2[:, 1])
    # plt.show()


    print("S")


    # # make output relative to the last observed frame
    # i_t = input_train[:, n_obs - 1, 0:2]
    # i_t = np.expand_dims(i_t, axis=1)
    # i_t = np.repeat(i_t, n_pred, axis=1)
    # output_train = output_train - i_t
    #
    # # reshape tensors
    # input_train = np.transpose(input_train, (1, 0, 2))
    # output_train = np.transpose(output_train, (1, 0, 2))
    #
    # input_train = torch.from_numpy(input_train).cuda()
    # output_train = torch.from_numpy(output_train).cuda()
    #
    # # plot_all(input_train)
    #
    #
    # # print(input_train.shape)
    # # # print(input_train[:, sample_pos, :])
    #
    # model = CI3PP(n_obs, n_pred).cuda()
    # model.load_state_dict(torch.load("./_out/weights/CI3PP_SUM/CI3PP_sum_60o_80p_2000e_4096b_2024-05-23_09-41-35.pth"))
    # model.eval()
    # pred = model(input_train[:, sample_pos, 0:2], input_train[:, sample_pos, 2:])
    # plot_rel(output_train[:, sample_pos, :2], pred)
    #
    #
    # m2p = M2P3(latent_dim=24, predict_frames=n_pred).cuda()
    # m2p.load_state_dict(torch.load("./_out/weights/M2P3/M2P360o_80p_250e_512b_2024-05-18_14-09-28.pth"))
    # m2p.eval()
    # pred_m2p = m2p.inference(input_train[:, sample_pos, 0:2])
    # plot_rel(output_train[:, sample_pos, :2], pred_m2p)



    # plot_path(input_train[:, sample_pos, :2], pred, output_train[:, sample_pos, :])



