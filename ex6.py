import numpy as np
import pandas as pd
import os
from optim.nonlinopt import nonlinopt # 非線形最適化
from selfmadeio.io_json import read_json
from selfmadeio.plot import make_plot, make_scatter # 作図
from selfmadeio.io_csv import save_csv
import random

# RMSEの計算
def calc_rmse(a, b):
    residual = a - b
    rmse = np.sqrt((residual*residual).sum()/len(residual))
    return rmse

# 目的関数
def objfun(param, x, y, gain=200):
    W      = calc_W(param, x, y, gain)
    y_calc = calc_y(W, param, x, gain)
    rmse   = calc_rmse(y_calc, y)
    loss = 0#relu(W[3])+relu(-W[1])#+relu(-W[2])#-sigmoid(param, 2, 10)*0.001+W[2]
    rmse_diff = calc_rmse(np.diff(y), np.diff(y_calc))
    fval = rmse + rmse_diff + loss
    return fval

def relu(x):
    return (x+abs(x))/2

# 制約式
def consfun(param):
    cons = [param[1]-1, 2-param[1]]
    return cons

# 重回帰分析における入力変数の情報を持つ行列の作成
def calc_Phi(x, param, gain):
    s     = sigmoid(x, param[0], gain)
    ones_ = np.ones_like(x)
    Phi = np.transpose(np.vstack([ones_*(1-s), x**param[1]*(1-s), ones_*s, x*s]))
    return Phi

# シグモイド関数
def sigmoid(x, x0, gain):
    return 1/(1+np.exp(-gain*(x-x0)))

# 線形回帰式による出力変数の計算
def calc_y(W, param, x, gain):
    Phi = calc_Phi(x, param, gain)
    Y = np.dot(Phi, W)
    return Y

# 制約付きリッジ回帰における未知係数の計算
def calc_W(param, x, y, gain, alpha=0.001, beta=1e3):
    Phi   = calc_Phi(x, param, gain)
    Phi_T = np.transpose(Phi)
    C = np.array([[1, param[0]**param[1], -1, -param[0]]]) # 等号制約
    C_T = np.transpose(C)
    KKT = np.vstack([ 2*np.dot(Phi_T, Phi)+alpha*np.eye(Phi.shape[1]), C ])
    KKT = np.hstack([ KKT, np.vstack([C_T, np.zeros([C.shape[0],C.shape[0]])]) ])
    KKT_inv = np.linalg.inv(KKT)
    aaa = np.transpose(np.hstack([ 2*np.dot(Phi_T, y), np.array([0])]))
    W_ = np.dot(KKT_inv, aaa)
    return W_[:Phi.shape[1]]


def main():
    # 設定ファイルの読み込み
    config = read_json("input/config_opt.json") 
    
    # 出力フォルダの設定
    output_path = os.getcwd() + '\output5'
    os.makedirs(output_path, exist_ok=True) # 出力フォルダがない場合は作る
    os.chdir(output_path)
    
    x_data = np.linspace(0, 1.5, 31)
    x_fit  = np.linspace(0, 1.5, 301)
    N_trial = 100
    for i in range(N_trial):
        # データの生成
        random.seed(i)
        param_true = [random.randint(-1, 201)/100] # 切り替わりのx
        random.seed(i+1000)
        param_true += [random.randint(100, 170)/100] # 次数
        random.seed(i+3000)
        
        W_true = [random.randint(-10, 10) for i in range(4)]
        W_true[1] = abs(W_true[1])
        W_true[3] = -abs(W_true[3])
        W_true[2] = W_true[0] + W_true[1]*param_true[0]**param_true[1] - W_true[3]*param_true[0]
        gain = 2000
        e=0
        if N_trial/2<=i:
            e=1
        random.seed(i+2000)
        y_data_pure = calc_y(W_true, param_true, x_data, gain)
        y_data = y_data_pure + np.array([random.randint(-100, 100)/150 for i in range(len(x_data))]) * e
        
        # パラメータの最適化
        l_param_init = np.linspace(0, 2, 6)
        fval=1e8
        for param_init in l_param_init:
            try:
                param_opt_, fval_, constype = nonlinopt(config, [param_init, 1.05], (x_data,y_data), objfun, consfun )
            except np.linalg.LinAlgError:
                param_opt_, fval_ = [param_init, 1.05], 1e10
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
