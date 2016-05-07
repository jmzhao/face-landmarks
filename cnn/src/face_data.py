
# coding: utf-8

# In[48]:

from skimage import data, io, filters
import matplotlib.pyplot as plt
import numpy as np
import os

# input image dimensions
img_chns, img_rows, img_cols = 3, 20, 20


# In[62]:

# helpers for reading data
def get_files(dirname) :
    f = []
    for (dirpath, dirnames, filenames) in os.walk(dirname):
        f.extend(filenames)
        break
    return f
def get_froot_list(dirname) :
    '''only read file root name with ".5pts"'''
    f = get_files(dirname)
    return sorted(filter(lambda x : len(x)>0, set(fn[:-5] for fn in f if fn[-5:]==".5pts")))

def is_froot_valid(froot) :
    return (os.path.isfile(froot+".png") 
            and io.imread(froot+".png").shape == (img_rows, img_cols, img_chns))

def read_pts(fname) :
    return np.array([[float(x) for x in line.split()] for line in open(fname)])

def calc_5_pts(pts) :
    return np.array([
        sum(pts[i] for i in (37,38,40,41)) / 4,
        sum(pts[i] for i in (43,44,46,47)) / 4,
        pts[30],
        pts[48],
        pts[54],
    ])[:,:2]


# In[68]:

# prepare data for CNN
subdirs = [
    '01_Indoor',   'afw',            'helen_trainset',  'lfpw_testset',
    '02_Outdoor',  'helen_testset',  'ibug',            'lfpw_trainset',
    'multipie'
]

def data(path, subdirs) :
    X = None; Y = None; l = None;
    for sub in subdirs :
        dirname = path + sub + "_20/" 
        l = list(filter(is_froot_valid, (dirname+f for f in get_froot_list(dirname))))
        n = [np.transpose(io.imread(frootname+".png"), (2,0,1)) for frootname in l]
        X = np.append(X, n, axis=0) if X is not None else np.array(n)
        n = [read_pts(frootname+".5pts") for frootname in l]
        Y = np.append(Y, n, axis=0) if Y is not None else np.array(n)
    X = X.astype('float32')
    Y = Y.astype('float32')
    X /= 255
    Y[:,:,0] /= img_rows
    Y[:,:,1] /= img_cols
    
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    
    return X, Y


# In[69]:

# split between train and test sets
def split_data(X, Y, ratio_train, rand_seed) :
    n = len(X)
    n_train = int(ratio_train * n)
    np.random.seed(rand_seed)
    ind = np.random.permutation(n)
    ind_train = ind[:n_train]
    ind_test = ind[n_train:]
    (X_train, Y_train) = X[ind_train], Y[ind_train]
    (X_test, Y_test) = X[ind_test], Y[ind_test]

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    return (X_train, Y_train), (X_test, Y_test)


def coord_to_corners(coord) :
    x, y = coord
    return np.array([(1-x)*(1-y), (1-x)*y, x*y, x*(1-y)])
def corners_to_coord(corners) :
    x0y0, x0y1, x1y1, x1y0 = corners
    return np.array([x1y1+x1y0, x1y1+x0y1]) / sum(corners)
def pointwise(array, ufunc) :
    return np.array([[ufunc(p) for p in pts] for pts in array])
