import numpy as np
import matplotlib.pyplot as plt

def init(folder):
    im_train_raw = np.load(folder + 'x_128_train.npy')
    im_valid_raw = np.load(folder + 'x_128_valid.npy')
    im_test_raw = np.load(folder + 'x_128_test.npy')
    lab_acr_train = np.load(folder + 'y_acrosome_train.npy')
    lab_acr_valid = np.load(folder + 'y_acrosome_valid.npy')
    lab_acr_test = np.load(folder + 'y_acrosome_test.npy')
    lab_head_train = np.load(folder + 'y_head_train.npy')
    lab_head_valid = np.load(folder + 'y_head_valid.npy')
    lab_head_test = np.load(folder + 'y_head_test.npy')
    lab_tail_train = np.load(folder + 'y_tail_train.npy')
    lab_tail_valid = np.load(folder + 'y_tail_valid.npy')
    lab_tail_test = np.load(folder + 'y_tail_test.npy')
    lab_vac_train = np.load(folder + 'y_vacuole_train.npy')
    lab_vac_valid = np.load(folder + 'y_vacuole_valid.npy')
    lab_vac_test = np.load(folder + 'y_vacuole_test.npy')
    lab_train = (lab_acr_train << 3)+(lab_head_train << 2)+(lab_tail_train << 1)+(lab_vac_train)
    lab_valid = (lab_acr_valid << 3)+(lab_head_valid << 2)+(lab_tail_valid << 1)+(lab_vac_valid)
    lab_test = (lab_acr_test << 3)+(lab_head_test << 2)+(lab_tail_test << 1)+(lab_vac_test)
    return im_train_raw, im_valid_raw, im_test_raw, \
           lab_train, lab_valid, lab_test

def draw(k, l):
    plt.figure()
    plt.grid(False)
    plt.xticks(range(16))
    thisplot = plt.bar(range(16), k, color="blue")
    plt.show()

def __init__():
    path = 'mhsma-dataset-master/mhsma/'

    im_train_raw, im_valid_raw, im_test_raw, \
    lab_train, lab_valid, lab_test = init(path)

    k = [0]*16
    for i in lab_train:
        k[i]+=1
    print(lab_train)
    l = len(lab_train)
    for i in range(16): print(str(k[i])+'('+str(i)+')', end=' ')
    print(f'\n{int(k[0]/l*100)}%')
    print(k[0]-max(k[1:]))
    k[0]=max(k[1:])
    #draw(k, l)

#__init__()
