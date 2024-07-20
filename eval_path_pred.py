from ped_path_predictor.CI3PP.ATT.train import CI3PP_ATT_Wrapper
from ped_path_predictor.CI3PP.ATT_BH.train import CI3PP_ATT_BH_Wrapper
from w_and_b.ATT_BH.train import CI3PP_ATT_BH_Wrapper as CI3PP_ATT_BH_Wrapper_WANDB
from ped_path_predictor.CI3PP.ATT_SH.train import CI3PP_ATT_SH_Wrapper
from ped_path_predictor.CI3PP.CVAE.train import CI3PP_CVAE_WRAPPER
from ped_path_predictor.CI3PP.CVAE_ATT.train import CI3PP_CVAE_ATT_WRAPPER
from P3VI.train import P3VIWrapper
from ped_path_predictor.m2p3 import PathPredictor
from prettytable import PrettyTable
data_paths = [
    './P3VI/data/single/01_int.npy',
    './P3VI/data/single/02_int.npy',
    './P3VI/data/single/03_int.npy',
    './P3VI/data/single/04_int.npy',
    './P3VI/data/single/05_int.npy',
    './P3VI/data/single/06_int.npy',
]

# data_paths = [
#     './P3VI/data/single/01_non_int.npy',
#     './P3VI/data/single/02_non_int.npy',
#     './P3VI/data/single/03_non_int.npy',
#     './P3VI/data/single/04_non_int.npy',
#     './P3VI/data/single/05_non_int.npy',
#     './P3VI/data/single/06_non_int.npy',
# ]

m2p3_mse = []
m2p3_fde = []

p3vi_mse = []
p3vi_fde = []


ci3pp_ATT_mse = []
ci3pp_ATT_fde = []

ci3pp_ATT_SH_mse = []
ci3pp_ATT_SH_fde = []

ci3pp_ATT_BH_mse = []
ci3pp_ATT_BH_fde = []

ci3pp_CVAE_mse = []
ci3pp_CVAE_fde = []

ci3pp_CVAE_ATT_mse = []
ci3pp_CVAE_ATT_fde = []



for p in data_paths:
    print(20*"#")
    print(p)

    # print("M2P3")
    # m2p3 = PathPredictor("./_out/weights/M2P3/M2P315o_20p_250e_512b_2024-05-18_14-09-28.pth",15, 20)
    # mse, fde = m2p3.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")

    print("OG M2P3")
    m2p3 = PathPredictor(
        "_out/weights/M2P3/M2P360o_80p_2000e_1024b_0.0001lr_2024-06-14_17-12-20.pth", 60, 80)
    mse, fde = m2p3.test(True, p)
    m2p3_mse.append(mse)
    m2p3_fde.append(fde)
    print(20 * "#", "\n")

    print("OG CI3P/P3VI")
    p3vi = P3VIWrapper(
        "_out/weights/P3VI/P3VI_60o_80p_2000e_1024b_5e-05lr_2024-06-14_17-12-10.pth", 60, 80)
    mse, fde = p3vi.test(True, p)
    p3vi_mse.append(mse)
    p3vi_fde.append(fde)
    print(20 * "#", "\n")


    # EARLY STOPPED AT 111
    print("CI3PP ATT")
    ci3P_att = CI3PP_ATT_Wrapper(
        "_out/weights/CI3P_ATT/CI3P_ATT60o_80p_2000e_1024b_5e-05lr_2024-06-13_18-25-34.pth", 60, 80)
    mse, fde = ci3P_att.test(p)
    ci3pp_ATT_mse.append(mse)
    ci3pp_ATT_fde.append(fde)
    print(20 * "#", "\n")


# EARLY STOPPED AT 105
    print("CI3PP ATT SH")
    ci3P_att = CI3PP_ATT_SH_Wrapper(
        "_out/weights/CI3P_ATT_SH/CI3P_ATT_SH60o_80p_2000e_1024b_5e-05lr_2024-06-13_22-06-43.pth", 60, 80)
    mse, fde = ci3P_att.test(p)
    ci3pp_ATT_SH_mse.append(mse)
    ci3pp_ATT_SH_fde.append(fde)
    print(20 * "#", "\n")

# EARLY STOPPED AT 182
    print("CI3PP ATT BH")
    ci3P_att = CI3PP_ATT_BH_Wrapper(
        "_out/weights/CI3P_ATT_BH/CI3P_ATT_BH60o_80p_2000e_1024b_5e-05lr_2024-06-13_18-25-46.pth", 60, 80)
    mse, fde = ci3P_att.test(p)
    ci3pp_ATT_BH_mse.append(mse)
    ci3pp_ATT_BH_fde.append(fde)
    print(20 * "#", "\n")

    # BANGER ATT BH
    print("CI3PP ATT BH BANGER")
    ci3P_att_BANGER = CI3PP_ATT_BH_Wrapper_WANDB(
        '_out/weights/CI3P_ATT_BH/CI3P_ATT_BH60o_80p_2000e_1024b_9.472028094027968e-05lr_256ed_16nh_2024-06-14_18-59-22.pth',
        60, 80, 256, 16)
    mse, fde = ci3P_att_BANGER.test(p)
    print(mse, fde)
    print(20 * "#", "\n")




    # EARLY STOPPED AT 195
    print("CI3PP CVAE")
    ci3P_cvae = CI3PP_CVAE_WRAPPER(
        "_out/weights/CI3P_CVAE/CI3P_CVAE60o_80p_2000e_1024b_5e-05lr_2024-06-13_18-25-09.pth", 60, 80)
    mse, fde = ci3P_cvae.test(True, p)
    ci3pp_CVAE_mse.append(mse)
    ci3pp_CVAE_fde.append(fde)
    print(20 * "#", "\n")


    # EARLY STOPPED AT 173
    print("CI3PP CVAE ATT")
    ci3P_cvae_att = CI3PP_CVAE_ATT_WRAPPER(
        "_out/weights/CI3PP_CVAE_ATT/CI3PP_CVAE_ATT60o_80p_2000e_1024b_5e-05lr_2024-06-13_18-25-34.pth", 60, 80)
    mse, fde = ci3P_cvae_att.test(True, p)
    ci3pp_CVAE_ATT_mse.append(mse)
    ci3pp_CVAE_ATT_fde.append(fde)
    print(20 * "#", "\n")









    # print("M2P3")
    # m2p3 = PathPredictor("./_out/weights/M2P3/M2P360o_80p_2000e_4096b_2024-05-24_18-44-58.pth", 60, 80)
    # mse, fde = m2p3.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("M2P3+")
    # m2p3p = M2P3P("./_out/weights/M2P3P/M2P3P60o_80p_2000e_4096b_2024-06-02_21-45-29.pth", 60, 80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    # #
    # #
    # print("M2P3+ L ATT_512")
    # m2p3p = M2P3P_att("./_out/weights/M2P3P_att_large/M2P3P_att_large60o_80p_2000e_512b_2024-06-06_09-58-16.pth", 60, 80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("M2P3+ L ATT_512:_LR00005")
    # m2p3p = M2P3P_att("./_out/weights/M2P3P_att_large_LR00005/M2P3P_att_large60o_80p_2000e_512b_2024-06-06_10-26-22.pth", 60,
    #                   80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("M2P3+ L ATT_4096")
    # m2p3p = M2P3P_att("./_out/weights/M2P3P_att_large/M2P3P_att_large60o_80p_2000e_4096b_2024-06-05_19-33-44.pth", 60,80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    #
    #
    # print("M2P3+ SUM ATT_512")
    # m2p3p = M2P3P_att_sum("./_out/weights/M2P3P_att_sum/M2P3P_att_sum60o_80p_2000e_512b_2024-06-06_09-57-32.pth", 60,
    #                   80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    #
    # print("M2P3+ SUM ATT_512")
    # m2p3p = M2P3P_att_sum("./_out/weights/M2P3P_att_sum_LR00005/M2P3P_att_sum60o_80p_2000e_512b_2024-06-06_10-24-22.pth", 60,
    #                       80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    #
    # print("M2P3+ SUM ATT_4096")
    # m2p3p = M2P3P_att_sum("./_out/weights/M2P3P_att_sum/M2P3P_att_sum60o_80p_2000e_4096b_2024-06-05_19-38-17.pth", 60,
    #                   80)
    # mse, fde = m2p3p.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    #
    #
    #
    # print("P3VI HIGH LR 4096")
    # p3vi = P3VIWrapper("./_out/weights/P3VI_HR/P3VI_HR60o_80p_2000e_4096b_2024-06-05_22-49-25.pth", 60, 80)
    # mse, fde = p3vi.test(True, p)
    # p3vi_mse_new.append(mse)
    # p3vi_fde_new.append(fde)
    # print(20 * "#", "\n")
    #
    # print("P3VI HIGH LR 512")
    # p3vi = P3VIWrapper("./_out/weights/P3VI_HR/P3VI_HR60o_80p_2000e_512b_2024-06-06_10-03-49.pth", 60, 80)
    # mse, fde = p3vi.test(True, p)
    # p3vi_mse_new.append(mse)
    # p3vi_fde_new.append(fde)
    # print(20 * "#", "\n")
    #
    # print("P3VI Low 00005 512")
    # p3vi = P3VIWrapper("./_out/weights/P3VI_LR_00005/P3VI_LR60o_80p_2000e_512b_2024-06-06_10-29-53.pth", 60, 80)
    # mse, fde = p3vi.test(True, p)
    # p3vi_mse_new.append(mse)
    # p3vi_fde_new.append(fde)
    # print(20 * "#", "\n")

    #
    # print("CI3P+")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP_SUM/CI3PP_sum_60o_80p_2000e_4096b_2024-05-23_09-41-35.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

    # print("CI3P+")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP/CI3PP60o_80p_250e_512b_2024-05-18_12-41-38.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("CI3P+ SUM HR 4096")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP_SUM/CI3PP_HIGH_LR_sum_60o_80p_2000e_4096b_2024-06-05_21-37-31.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

    # print("CI3P+ SUM HR 512")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP_SUM/CI3PP_sum_60o_80p_2000e_512b_2024-06-06_10-04-28.pth",
    #                            60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("CI3P+ SUM LR 512")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP_SUM_LR_00005/CI3PP_sum_60o_80p_2000e_512b_2024-06-06_10-21-03.pth",
    #                            60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")




    #
    # print("CI3P+ HIGH LR")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP_SUM/CI3PP_HIGH_LR_sum_60o_80p_2000e_4096b_2024-06-05_21-37-31.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

    # print("CI3P+ BEF ENC")
    # ci3pp = CI3PPWrapper_BEF_ENC("./_out/weights/CI3PP_BEF_ENC/CI3PP_bef_enc_60o_80p_2000e_4096b_2024-06-02_22-40-55.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

    # print("P3VI")
    # p3vi = P3VIWrapper("./_out/weights/P3VI/P3VI_15o_20p_250e_512b_2024-05-18_12-49-33.pth", 15, 20)
    # mse, fde = p3vi.test(True, p)
    # p3vi_mse_new.append(mse)
    # p3vi_fde_new.append(fde)
    # print(20 * "#", "\n")




    # print("CI3P+ SMALL")
    # ci3pp = CI3PPWrapper_LESS_HEADS("./_out/weights/CI3PP_LESS_HEADS/CI3PP_less_head_60o_80p_2000e_4096b_2024-05-26_17-50-43.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("CI3P+ SMALL LOW LR")
    # ci3pp = CI3PPWrapper_LESS_HEADS(
    #     "./_out/weights/CI3PP_LESS_HEADS/CI3PP_less_head_lowlr_60o_80p_2000e_4096b_2024-05-30_14-56-55.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("CIPT")
    # cipt = CIPTWrapper(
    #     "./_out/weights/CIPT/CIPT60o_80p_2000e_4096b_2024-05-30_16-29-49.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

# print("CI3P+ LOW LR SUM")
    # ci3pp = CI3PPWrapper_60_80("./_out/weights/CI3PP_SUM_LOW_LR/CI3PP_sum_60o_80p_2000e_4096b_2024-05-23_12-39-20.pth", 60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")
    # #
    # print("CI3P+ POS ENC SUM")
    # ci3pp = CI3PPWrapper_POS_ENC_60_80("./_out/weights/pos_enc_sum/CI3PP60o_80p_2000e_4096b_05lr_2024-05-23_09-39-52.pth",
    #                            60, 80)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

    # print("CI3P+")
    # ci3pp = CI3PPWrapper("./_out/weights/CI3PP/CI3PP15o_20p_250e_512b_2024-05-18_12-38-39.pth", 15, 20)
    # mse, fde = ci3pp.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")



    # print("CI3P+ NO ENC")
    # ci3pp_no_enc = CI3PPWrapper_NO_ENC("./_out/weights/CI3PP_no_enc/CI3PP_no_enc15o_20p_250e_512b_2024-05-18_12-45-30.pth", 15, 20)
    # mse, fde = ci3pp_no_enc.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")

    # print("CI3P+ NO ENC")
    # ci3pp_no_enc = CI3PPWrapper_NO_ENC_60_80(
    #     "./_out/weights/CI3PP_no_enc/CI3PP_no_enc60o_80p_250e_512b_2024-05-18_12-45-54.pth", 60, 80)
    # mse, fde = ci3pp_no_enc.test(p)
    # ci3pp_mse.append(mse)
    # ci3pp_fde.append(fde)
    # print(20 * "#", "\n")


    # print("BiTrap-relative")
    # bitrap = BiTrapWrapper(model_path="./_out/weights/bitrap_relative_100e_512b_60obs_80pred_2024-05-12_17-56-33.pth", observed_frame_num=60, predicting_frame_num=80)
    # mse, fde = bitrap.test(p)
    # bitrap_mse.append(mse)
    # bitrap_fde.append(fde)
    # print(20 * "#", "\n")

    # print("BiTrap-relative")
    # bitrap = BiTrapWrapper(model_path="./_out/weights/old/bitrap_relative_100_512_15_20.pth", observed_frame_num=15, predicting_frame_num=20)
    # mse, fde = bitrap.test(p)
    # bitrap_mse.append(mse)
    # bitrap_fde.append(fde)
    # print(20 * "#", "\n")


# print("BiTrap-absolute")
    # bitrap = BiTrapWrapperAbsolute(model_path="./_out/weights/bitrap_absolute_100e_512b_60obs_80pred_2024-05-12_17-56-33.pth", observed_frame_num=60,
    #                        predicting_frame_num=80)
    # mse, fde = bitrap.test(p)
    # bitrap_mse.append(mse)
    # bitrap_fde.append(fde)
    # print(20 * "#", "\n")


    # print("P3VI")
    # p3vi = P3VIWrapper("./_out/weights/new_2000_256_all_seed_0_p3vi_best_15_20.pth",60,20)
    # mse, fde = p3vi.test(True,p)
    # p3vi_mse.append(mse)
    # p3vi_fde.append(fde)
    # print(20*"#","\n")

# string = ""
# for mse,fde in zip(m2p3_mse, m2p3_fde):
#     string += "M2P3 " + str(mse) +"/"+str(fde)
# print(string)
# #
# string = ""
# for mse,fde in zip(p3vi_mse_new, p3vi_fde_new):
#     string += "P3VI new" + str(mse) +"/"+str(fde)
# print(string)
#
# string = ""
# for mse,fde in zip(ci3pp_mse, ci3pp_fde):
#     string += ("CI3P+ ") + str(mse) +"/"+str(fde)
# print(string)

# string = ""
# for mse,fde in zip(p3vi_mse, p3vi_fde):
#     string += "P3VI stock" + str(mse) +"/"+str(fde)
# print(string)


rows = ["Model"]
m2p3_row = ["M2P3"]
p3vi_row = ["P3VI"]
ci3pp_ATT_row = ["CI3PP_ATT"]
ci3pp_ATT_SH_row = ["CI3PP_ATT_SH"]
ci3pp_ATT_BH_row = ["CI3PP_ATT_BH"]
ci3pp_CVAE_row = ["CI3PP_CVAE"]
ci3pp_CVAE_ATT_row = ["CI3PP_CVAE_ATT"]

for i in range(len(data_paths)):
    rows.append(f"Scenario {i+1}")
    m2p3_row.append(f"{m2p3_mse[i]}/{m2p3_fde[i]}")
    p3vi_row.append(f"{p3vi_mse[i]}/{p3vi_fde[i]}")
    ci3pp_ATT_row.append(f"{ci3pp_ATT_mse[i]}/{ci3pp_ATT_fde[i]}")
    ci3pp_ATT_SH_row.append(f"{ci3pp_ATT_SH_mse[i]}/{ci3pp_ATT_SH_fde[i]}")
    ci3pp_ATT_BH_row.append(f"{ci3pp_ATT_BH_mse[i]}/{ci3pp_ATT_BH_fde[i]}")
    ci3pp_CVAE_row.append(f"{ci3pp_CVAE_mse[i]}/{ci3pp_CVAE_fde[i]}")
    ci3pp_CVAE_ATT_row.append(f"{ci3pp_CVAE_ATT_mse[i]}/{ci3pp_CVAE_ATT_fde[i]}")

t = PrettyTable(rows)
t.add_row(m2p3_row)
t.add_row(p3vi_row)
t.add_row(ci3pp_ATT_row)
t.add_row(ci3pp_ATT_SH_row)
t.add_row(ci3pp_ATT_BH_row)
t.add_row(ci3pp_CVAE_row)
t.add_row(ci3pp_CVAE_ATT_row)

print(t)
