import numpy as np
import pandas as pd
import os
from optim.nonlinopt import nonlinopt # 非線形最適化
from selfmadeio.io_json import read_json
from selfmadeio.plot import make_plot, make_scatter # 作図
from selfmadeio.io_csv import save_csv
import random

# 目的関数
def objfun(param,x,y, gain=200):
    W      = calc_W(param, x, y, gain)
    y_calc = calc_y(W, param, x, gain)
    residual = y_calc - y
    rmse = np.sqrt((residual*residual).sum()/len(residual))
    return rmse

# 制約式
def consfun(param):
    cons = []
    return cons

# 重回帰分析における入力変数の情報を持つ行列の作成
def calc_Phi(x, param, gain):
    s     = sigmoid(x, param, gain)
    ones_ = np.ones_like(x)
    Phi = np.transpose(np.vstack([ones_*(1-s), x*(1-s), ones_*s, x*s]))
    return Phi
    
# シグモイド関数
def sigmoid(x, x0, gain):
    return 1/(1+np.exp(-gain*(x-x0)))

# 回帰
def calc_y(W, param, x, gain):
    Phi = calc_Phi(x, param, gain)
    Y = np.dot(Phi, W)
    return Y

# リッジ回帰における未知係数の計算
def calc_W(param, x, y, gain, alpha=0.02):
    Phi   = calc_Phi(x, param, gain)
    Phi_T = np.transpose(Phi)
    A = np.linalg.pinv( np.dot(Phi_T, Phi)+alpha*np.eye(Phi.shape[1]) )
    B = np.dot(A, Phi_T)
    W = np.dot(B, y)
    return W


def main():
    # 設定ファイルの読み込み
    config = read_json("input/config_opt.json") 
    
    # 出力フォルダの設定
    output_path = os.getcwd() + '\output1'
    os.makedirs(output_path, exist_ok=True) # 出力フォルダがない場合は作る
    os.chdir(output_path)
    
    x_data = np.linspace(0, 1.5, 31)
    x_fit  = np.linspace(0, 1.5, 301)
    N_trial = 100
    for i in range(N_trial):
        # データの生成
        random.seed(i)
        W_true = [random.randint(-10, 10) for i in range(4)]
        random.seed(i+1000)
        param_true = random.randint(-1, 101)/100
        gain = 2000
        e=0
        if N_trial/2<=i:
            e=1
        random.seed(i+2000)
        y_data_pure = calc_y(W_true, param_true, x_data, gain)
        y_data = y_data_pure + np.array([random.randint(-100, 100)/150 for i in range(len(x_data))]) * e
        
        # パラメータの最適化
        l_param_init = np.linspace(0, 1, 21)
        fval=1e8
        for param_init in l_param_init:
            param_opt_, fval_, constype = nonlinopt(config, param_init, (x_data,y_data), objfun, consfun )
            if fval>fval_:
                param_opt = param_opt_
                fval = fval_
        W_opt = calc_W(param_opt, x_data, y_data, gain)
        
        # フィッティング
        gain = 200
        y_fit       = calc_y(W_opt,  param_opt,  x_fit, gain)
        y_data_pure = calc_y(W_true, param_true, x_fit, gain)
        
        # 出力
        """
        print("x0: true={:.2f}, opt={:.2f}".format(param_true, param_opt[0]))
        print("W_true")
        print(W_true)
        print("W_opt")
        print(W_opt)
        """
        
        fig_name = "fitting_"+str(i+1).zfill(3)
        make_plot([x_data, x_fit, x_fit], [y_data, y_fit, y_data_pure], fig_name,\
                  marker=["o", "None", "None"], line_style=["None", "-", "--"], color=["k", "r", "b"])

        
main()
