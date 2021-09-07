import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mg

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文


def visualizedTrainAndTest(log_file, image_save_LossAcc, image_save_PRF, data_range_dict):
    '''
    绘制训练过程各指标
    @param log_file: 保存训练过程的文本文件
    @param image_save: 保存图形文件
    @param data_range_dict: {"TRAIN":train_data_name,"TEST0":test_data_name0,...}
    @return:
    '''

    res = dict()
    with open(log_file, "r", encoding="utf-8") as fr:
        for line in fr:
            for k in data_range_dict.keys():
                if k in line:
                    line = line.strip().split("\t")[1:]
                    currentData = [float(x) for x in line]
                    if k not in res.keys():
                        res[k] = [currentData]
                    else:
                        res[k].append(currentData)

    for k in res.keys():
        res[k] = np.array(res[k])

    number = len(res.keys())

    linewidth = 10
    pad = 100
    labelpad = 70
    fontdict = {'fontsize': 200}
    prop = {'size': 150}
    x_range = range(1, np.shape(res["TRAIN"])[0] + 1)
    xticks_range = range(0, np.shape(res["TRAIN"])[0] + 1, 10)
    xticks_fontsize = 40
    yticks_fontsize = 100
    lengend_loc = 'lower right'

    plt.figure(figsize=(200, 100))
    plt.subplot(1, 2, 1)
    plt.plot(x_range, res["TRAIN"][:, 0], linewidth=linewidth)
    plt.xlabel("Epoch", fontdict=fontdict, labelpad=labelpad)
    plt.title("Loss", fontdict=fontdict, pad=pad)
    plt.xticks(xticks_range, fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x_range, res["TRAIN"][:, 1], linewidth=linewidth)
    plt.xlabel("Epoch", fontdict=fontdict, labelpad=labelpad)
    plt.title("Accuracy", fontdict=fontdict, pad=pad)
    plt.xticks(xticks_range, fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.grid()

    plt.suptitle("Loss and Acc on %s" % data_range_dict["TRAIN"], fontsize=fontdict["fontsize"])
    plt.savefig(image_save_LossAcc)
    plt.close()

    plt.figure(figsize=(200, 100))
    plt.subplot(1, number, 1)
    plt.plot(x_range, res["TRAIN"][:, -3], 'r-', linewidth=linewidth)
    plt.plot(x_range, res["TRAIN"][:, -2], 'g-', linewidth=linewidth)
    plt.plot(x_range, res["TRAIN"][:, -1], 'b-', linewidth=linewidth)
    plt.xlabel("Epoch", fontdict=fontdict, labelpad=labelpad)
    plt.ylim([0, 100])
    plt.title(data_range_dict["TRAIN"], fontdict=fontdict, pad=pad)
    plt.legend(["Precision", "Recall", "F$_1$"], prop=prop, loc=lengend_loc)
    plt.xticks(xticks_range, fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.grid()

    for l in range(number - 1):
        print("EPOCHS: ", np.shape(res["TRAIN"])[0], " ", data_range_dict["TEST%d" % l])
        if l == number - 2:
            psc_max = 0.0
            rsc_max = 0.0
            psc_current = 0.0
            rsc_current = 0.0
            kmax = 0

        for k in range(len(res["TEST%d" % l])):
            if l == number - 2:
                psc_current = res["TEST%d" % l][k][-3]
                rsc_current = res["TEST%d" % l][k][-2]
                if rsc_current > rsc_max:
                    rsc_max = rsc_current
                    psc_max = psc_current
                    kmax = k + 1
                elif rsc_current == rsc_max:
                    if psc_current > psc_max:
                        psc_max = psc_current
                        kmax = k + 1

            # print("%3d--> &" % (k + 1) + " &".join(["%.1f" % t for t in res["TEST%d" % l][k]]))
            print("%3d--> " % (k + 1) + " ".join(["%.1f" % t for t in res["TEST%d" % l][k]]))

        print()
        if l == number - 2:
            print(kmax, " --> ", " ".join(["%.1f" % t for t in res["TEST%d" % l][kmax - 1]]))
            print(kmax, " --> &" + "&".join(["%.1f" % t for t in res["TEST%d" % l][kmax - 1]]))

        print()

        plt.subplot(1, number, l + 2)
        plt.plot(x_range, res["TEST%d" % l][:, -3], 'r-', linewidth=linewidth)
        plt.plot(x_range, res["TEST%d" % l][:, -2], 'g-', linewidth=linewidth)
        plt.plot(x_range, res["TEST%d" % l][:, -1], 'b-', linewidth=linewidth)
        plt.xlabel("Epoch", fontdict=fontdict, labelpad=labelpad)
        plt.ylim([0, 100])
        plt.title(data_range_dict["TEST%d" % l], fontdict=fontdict, pad=pad)
        plt.legend(["Precision", "Recall", "F$_1$"], prop=prop, loc=lengend_loc)
        plt.xticks(xticks_range, fontsize=xticks_fontsize)
        plt.yticks(fontsize=yticks_fontsize)
        plt.grid()

    print()

    plt.suptitle("Precision, Recall, F$_1$", fontsize=fontdict["fontsize"])
    plt.savefig(image_save_PRF)
    plt.close()


if __name__ == "__main__":
    data_range_dict = {
        "TRAIN": "271k_train",
        "TEST0": "sighan_test",
    }
    visualizedTrainAndTest(
        "result/REALISE_ACUM_271k.log",
        "result/REALISE_ACUM_271k_LossAcc.jpg",
        "result/REALISE_ACUM_271k_PRF.jpg",
        data_range_dict
    )
    # data_range_dict = {
    #     "TRAIN": "271k_train",
    #     "TEST0": "sighan_test",
    # }
    # visualizedTrainAndTest(
    #     "result/CBERT_ACUM_271k.log",
    #     "result/CBERT_ACUM_271k_LossAcc.jpg",
    #     "result/CBERT_ACUM_271k_PRF.jpg",
    #     data_range_dict
    # )

    data_range_dict = {
        "TRAIN": "sighan_train",
        # "TEST0": "sighan_test_wrong",
        "TEST0": "sighan_test",
    }

    # visualizedTrainAndTest(
    #     "result/REALISE_C_ACUM_sighan.log",
    #     "result/REALISE_C_ACUM_sighan_LossAcc.jpg",
    #     "result/REALISE_C_ACUM_sighan_PRF.jpg",
    #     data_range_dict
    # )

    # visualizedTrainAndTest(
    #     "result/CBERT_ACUM_sighan.log",
    #     "result/CBERT_ACUM_sighan_LossAcc.jpg",
    #     "result/CBERT_ACUM_sighan_PRF.jpg",
    #     data_range_dict
    # )
    # visualizedTrainAndTest(
    #     "result/SMCBERT_ACUM_sighan.log",
    #     "result/SMCBERT_ACUM_sighan_LossAcc.jpg",
    #     "result/SMCBERT_ACUM_sighan_PRF.jpg",
    #     data_range_dict
    # )
    # visualizedTrainAndTest(
    #     "result/SMCBERT_ACUM_sighan.log",
    #     "result/SMCBERT_ACUM_sighan_LossAcc.jpg",
    #     "result/SMCBERT_ACUM_sighan_PRF.jpg",
    #     data_range_dict
    # )
    visualizedTrainAndTest(
        "result/REALISE_ACUM_sighan.log",
        "result/REALISE_ACUM_sighan_LossAcc.jpg",
        "result/REALISE_ACUM_sighan_PRF.jpg",
        data_range_dict
    )
