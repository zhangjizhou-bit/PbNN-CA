# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
#
# 文件 : main.py
# 说明 : 主程序入口
# 时间 : 2021/03/29
# 作者 : 张继洲

import os
from train import FPM_reconstruction
from utility import *
from parameters import args
from parameters_zuo import args2
from parameters_zheng import args3
from parameters_tian import args4

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    if args.mode == "simulator":
        # simulate_data(args)
        simulate_data_star(args)
    elif args.mode == "real_data":
        load_real_data(args)

    args.mode = "estimator"
    fpm_reconstruction = FPM_reconstruction(args)
    fpm_reconstruction.train_model()
    fpm_reconstruction.eval_model()

    Metrics(args)
    ShowResults(args)


def main2():

    load_real_data2(args2)

    args2.mode = "estimator"
    fpm_reconstruction = FPM_reconstruction(args2)
    fpm_reconstruction.train_model()
    fpm_reconstruction.eval_model()

    Metrics(args2)
    ShowResults(args2)


def main3():

    load_real_data3(args3)

    args3.mode = "estimator"
    fpm_reconstruction = FPM_reconstruction(args3)
    fpm_reconstruction.train_model()
    fpm_reconstruction.eval_model()

    Metrics(args3)
    ShowResults(args3)


def main4():

    load_real_data4(args4)

    args4.mode = "estimator"
    fpm_reconstruction = FPM_reconstruction(args4)
    fpm_reconstruction.train_model()
    fpm_reconstruction.eval_model()

    Metrics(args4)
    ShowResults(args4)



if __name__ == '__main__':
    main()
    # main2()
    # generate_cos4_zuo(args2)

