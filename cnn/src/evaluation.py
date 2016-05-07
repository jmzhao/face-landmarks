
# coding: utf-8

# In[125]:

import numpy as np
import face_data as fd
import logging
import time

logging.basicConfig(level=logging.INFO)


# In[149]:

learned_models = [
    {
        'model_name' : "M1",
        'model_fname_root' : "all(ratio-0.8,rand-1337),20px,500ep",
        'description' : "the basic one",
        'compile_info' : {
            'loss' : 'mse',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
        'pred_converter' : lambda Y_pred : Y_pred.reshape((Y_pred.shape[0], 5, 2)),
    },
    {
        'model_name' : "M2",
        'model_fname_root' : "all(ratio-0.8,rand-1337),one-more-Dense-refined,20px,500ep",
        'description' : "one-more-Dense-refined",
        'compile_info' : {
            'loss' : 'mse',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
    },
    {
        'model_name' : "M2'",
        'model_fname_root' : "all(ratio-0.8,rand-1337),one-more-Dense,20px,500ep",
        'description' : "one-more-Dense",
        'compile_info' : {
            'loss' : 'mse',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
    },
    {
        'model_name' : "M3",
        'model_fname_root' : "all(ratio-0.8,rand-1337),3sub-parallel,20px,500ep",
        'description' : "3submodel",
        'compile_info' : {
            'loss' : 'mse',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
    },
    {
        'model_name' : "M4",
        'model_fname_root' : "all(ratio-0.8,rand-1337),single-mask_single,20px,500ep",
        'description' : "single-mask_single",
        'compile_info' : {
            'loss' : 'mse',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
    },
    {
        'model_name' : "M4'",
        'model_fname_root' : "all(ratio-0.8,rand-1337),masked,20px,500ep",
        'description' : "masked",
        'compile_info' : {
            'loss' : 'mse',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
    },
    {
        'model_name' : "M1-distr",
        'model_fname_root' : "all(ratio-0.8,rand-1337),corner128,20px,500ep",
        'description' : "corner-128",
        'compile_info' : {
            'loss' : 'categorical_crossentropy',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
        'pred_converter' : lambda Y_pred : fd.pointwise(Y_pred, fd.corners_to_coord),
    },
]
learned_models.extend([
    {
        'model_name' : "M1'-distr",
        'model_fname_root' : "corners-all(ratio-0.8,rand-1337),20px,500ep",
        'description' : "corner-256",
        'compile_info' : {
            'loss' : 'categorical_crossentropy',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
        'pred_converter' : lambda Y_pred : learned_models[6]['pred_converter'](np.array(Y_pred).transpose((1,0,2))),
    },
    {
        'model_name' : "M4-distr",
        'model_fname_root' : "all(ratio-0.8,rand-1337),corner,single-mask_single,20px,500ep",
        'description' : "corner,single-mask_single",
        'compile_info' : {
            'loss' : 'categorical_crossentropy',
            'optimizer' : 'adadelta',
            'metrics' : ['accuracy'],
        },
        'pred_converter' : learned_models[6]['pred_converter'],
    },
])
data_points = [
    {
        'point_id' : 0,
        'point_name' : 'LE',
    },
    {
        'point_id' : 1,
        'point_name' : 'RE',
    },
    {
        'point_id' : 2,
        'point_name' : 'N',
    },
    {
        'point_id' : 3,
        'point_name' : 'LM',
    },
    {
        'point_id' : 4,
        'point_name' : 'RM',
    },
]
def calc_error_one_point(p_truth, p_pred) :
    return np.sqrt(sum((p_truth - p_pred) ** 2))
supported_measures = [
    {
        'name' : 'error',
        'func' : calc_error_one_point,
        'stat' : np.mean,
    },
    {
        'name' : 'accuracy005',
        'func' : lambda pt, pp : calc_error_one_point(pt, pp) < 0.05,
        'stat' : np.mean,
    },
    {
        'name' : 'accuracy010',
        'func' : lambda pt, pp : calc_error_one_point(pt, pp) < 0.10,
        'stat' : np.mean,
    },
]


# In[27]:

def load_model(dir_path, model_fname_root, compile_info, **kwargs) :
    model_root = dir_path + model_fname_root
    import keras.models
    model2 = keras.models.model_from_json(open(model_root+'.json').read())
    model2.load_weights(model_root+'.weight')
    model2.compile(**compile_info)
    return model2


# In[150]:

def evaluate_one_point(P_truth, P_pred, measures=supported_measures) :
    info = dict()
    for m_info in measures :
        A = [m_info['func'](p_truth, p_pred) for p_truth, p_pred in zip(P_truth, P_pred)]
        a = m_info['stat'](A)
        info[m_info['name']] = a 
    return info
def evaluate(Y, Y_pred, pred_converter=None) :
    """given information of a learned model, return evaluation info"""
    if pred_converter is not None :
        Y_pred = pred_converter(Y_pred)
    evaluation = [
        (data_point_info['point_name'],
         evaluate_one_point(Y[:,data_point_info['point_id']], Y_pred[:,data_point_info['point_id']])) 
        for data_point_info in data_points
    ]
    return evaluation


# In[28]:




# In[67]:

# load data
logging.info("loading data from: " + " ".join(fd.subdirs))
X, Y = fd.data('../../../result_20/', fd.subdirs)
(X_train, Y_train), (X_test, Y_test) = fd.split_data(X, Y, ratio_train=0.8, rand_seed=1337)


# In[151]:

for model_info in learned_models :
    logging.info("evaluating model %s (%s)"%(model_info['model_name'], model_info['description']))
    print(model_info['model_name'])
    print(model_info['description'])
    model = load_model(dir_path='../model/', **model_info)
    t0 = time.clock()
    for x_test in X_test :
        y_pred = model.predict(np.array([x_test]))
    print('time (predict one-by-one):', time.clock() - t0)
    t0 = time.clock()
    Y_pred = model.predict(X_test)
    print('time (predict all-at-once):', time.clock() - t0)
    evaluation = evaluate(Y_test, Y_pred, model_info.get('pred_converter'))
    print(*evaluation, sep='\n')


# In[ ]:



