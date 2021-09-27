import numpy as np
import sys, os
import re

from pdb import set_trace as st

CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]

KNN_PARAMS = ['10','20','100','200']
LINCLS_PARAMS = ['@1', '@5']

def compute_mce(corruption_accs):
    """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
    mce = 0.
    ALEXNET_ERR = [
        0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
        0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
        0.606500
    ]
    for i in range(len(CORRUPTIONS)):
        avg_err = 100 - np.mean(corruption_accs[CORRUPTIONS[i]],axis=1)
        ce = avg_err / ALEXNET_ERR[i]
        mce += ce / 15
    return mce

def analyze_knn_log(log_name):
    with open(log_name) as f:
        txt = f.read()
    m = re.findall(r'.+ Top1: (\d+\.\d+), Top5: (\d+\.\d+)', txt)
    cnt = 0
    mCE_list = np.zeros(4)
    corruption_accs = {}
    for c in CORRUPTIONS:
        # print(c)
        ACC = np.zeros([4,5])
        for s in range(5):
            for i in range(4):
                ACC[i,s] = float(m[cnt][0])
                cnt += 1
        corruption_accs[c] = ACC
        # for k in range(4):
        #     print(f"{KNN_PARAMS[k]}-NN classifier result: mCE: {mCE[k]}")
        # mCE_list += mCE
    
    mces = compute_mce(corruption_accs)

    for k in range(4):
            print(f"Lincls{KNN_PARAMS[k]} classifier result: total mCE: {mces[k]}")

def analyze_lincls_log(log_name):
    with open(log_name) as f:
        txt = f.read()
    m = re.findall(r'.+Acc@1 (\d+\.\d+) Acc@5 (\d+\.\d+)', txt)
    st()
    cnt = 0
    mCE_list = np.zeros(2)
    corruption_accs = {}
    try:
    	for c in CORRUPTIONS:
		    # print(c)
            ACC = np.zeros([2,5])
            for s in range(5):
                for i in range(2):
                ACC[i,s] = float(m[cnt][0])
                cnt += 1
            corruption_accs[c] = ACC
            # for k in range(4):
            #     print(f"{KNN_PARAMS[k]}-NN classifier result: mCE: {mCE[k]}")
            # mCE_list += mCE
    except:
    	st()
    mces = compute_mce(corruption_accs)

    for k in range(2):
            print(f"{KNN_PARAMS[k]}-NN classifier result: total mCE: {mces[k]}")


if __name__ == '__main__':
    log_name = sys.argv[1]
    #analyze_knn_log(log_name)
    analyze_lincls_log(log_name)
