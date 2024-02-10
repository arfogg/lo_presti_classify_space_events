import pandas as pd
import numpy as np
from kymatio.torch import Scattering1D
import torch
from torch.nn import Linear, NLLLoss, LogSoftmax, Sequential
from torch.optim import Adam
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold, GridSearchCV
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

####################### DATA PREPARATION #######################

def fill_NaN(sequence_dataframe: pd.core.series.Series, filling_method: str) -> np.ndarray:
    '''
    Fill missing values in a time series 
    according to passed method (linear_interpolation or mean),
    and return the filled sequence as a numpy array.
    '''
    if filling_method == 'linear_interpolation': # linear interpolation
        sequence_dataframe = sequence_dataframe.interpolate(method='linear')
        # fill NaN at the beginning or end of the series, if any
        sequence_dataframe.ffill(inplace=True) # forward filling
        sequence_dataframe.bfill(inplace=True) # backward filling

    elif filling_method == 'mean':
        mean_value = sequence_dataframe.mean()
        sequence_dataframe = sequence_dataframe.fillna(mean_value) # mean 
    
    return np.array(sequence_dataframe)


####################### EXTRACT FEATURES #######################

def compute_Scattering1D(sequences: torch.Tensor, T: int, J: int, Q:int, log_eps: float) -> torch.Tensor:
    '''
    Compute Scattering1D to extract features from the passed time series sequences 

    Input:
        -   sequences: matrix with as many rows as the time series observed, and as many columns as the time steps 
        -   T, J, Q: parameters for the scattering transform:
            T is the num of samples, given by the size of the input
            J is the max scale of the wavelet transform, specified as a power of 2
            Q is the num of wavelets per octave (freq at resolution 1/Q)
        -   log_eps: small costant to be added to the scattering coefficients before computing
                    the log, to prevent obtaining very large values when the coefficients are close 
                    to zero

    Returns:
        -   Sc_all: matrix containing the features extracted from each sequence;
                    each row in the matrix correspond to the features extracted from 
                    the sequence at the same row index in the "sequences" matrix;
                    each column represent a feature
    '''
    scattering = Scattering1D(J, T, Q)

    # compute scattering transform for all signals 
    Sc_all = scattering.forward(sequences)
    Sc_all = Sc_all[:,1:,:] # remove 0th order (low-pass filtered version of the signals, not carring meaningful info)

    # take log (log-scatteing transform)
    Sc_all = torch.log(torch.abs(Sc_all) + log_eps)

    # average time dimension (last dim) to get time-shift invariant representation
    Sc_all = torch.mean(Sc_all, dim=-1) # equivalent to Global Avg Pooling 

    return Sc_all




####################### MODELS #######################


def get_logistic_model(num_input, num_classes):
    '''
    Logistic regression

    Inputs:
        -   num_input: number of time steps or number of features per sequence (num columns)
        -   num_classes: number of classes

    Return:
        -  model, optimizer, criterion: Sequential model, optimizer to be used, and criterion to optimized 
    '''
    # fully connected linear layer + log softmax
    model = Sequential(Linear(num_input, num_classes), LogSoftmax(dim=1))
    optimizer = Adam(model.parameters())
    criterion = NLLLoss() # negative log likelihood loss
    return model, optimizer, criterion

def train_logistic_model(Sx_train: torch.Tensor, y_train: torch.Tensor,
                         model, optimizer, criterion, 
                         num_epochs, n_samples, batch_size, 
                         verbose = True, random_seed = 2):
    
    '''
    Train model

    Input:
        -   Sx_train: matrix of features, or raw sequences
        -   y_train: vector of labels
        -   model, optimizer, criterion: Sequential model, optimizer to be used, and criterion to optim
        -   num_epochs, batch_size: number of epochs to be trained over, and batch size
        -   n_samples: number of sequences in the training set (rows matrix)
        -   verbose (bool): whether to print or not training process

    '''

    n_batches = n_samples // batch_size

    for e in range(num_epochs):
        # random permutation
        torch.manual_seed(random_seed)
        perm = torch.randperm(n_samples)

        # for each batch, compute gradient wrt loss
        for i in range(n_batches):
            idx = perm[i * batch_size : (i+1) * batch_size]
            model.zero_grad()
            resp = model.forward(Sx_train[idx])
            loss = criterion(resp, y_train[idx])
            loss.backward()
            # take one step
            optimizer.step()

        
        resp = model.forward(Sx_train)
        avg_loss = criterion(resp, y_train)

        # predict training set classes to compute accuracy while training
        y_hat = resp.argmax(dim=1)
        accuracy = (y_train == y_hat).float().mean()

        if verbose:
            print('Epoch {}, average loss = {:1.3f}, accuracy = {:1.3f}'.format(e, avg_loss, accuracy))
    

def SVC_params_opt(Sx_train, y_train,
                   n_splits = 10, 
                   n_repeats = 5, 
                   param_grid = { 
                                'C': [ 0.1,  1. , 10. ], # regularization
                                'class_weight':[{1:2, 0:1},  # give weight 2 to class 1 (event)
                                                {1:1, 0:.5}],
                                'kernel': ['poly', 'rbf', 'sigmoid'],
                                'gamma': ['scale', 'auto'] # kernel coefficent for poly, rbf, sigmoid
                                }):
    '''
    Perform cross-validation on support vector classifier parameters,
    and return a dictionary with the optimized parameters
    '''
    # default: 10 folds CV

    # SVC parameters:
    # ['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 
    # 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose']

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    grid_search = GridSearchCV(estimator=SVC(), 
                                param_grid=param_grid, 
                                scoring='accuracy', 
                                refit='accuracy', 
                                n_jobs=-1, 
                                cv=cv)
    gridCV = grid_search.fit(Sx_train, y_train)
    return gridCV.best_params_


####################### MODELS EVALUATION #######################

def plot_confusion_matrix(confusion):
    plt.figure()
    plt.imshow(confusion, cmap='viridis') 

    tick_locs = np.arange(2)
    ticks = ['{}'.format(i) for i in range(0, 2)]

    plt.xticks(tick_locs, ticks)
    plt.yticks(tick_locs, ticks)

    plt.ylabel('True')
    plt.xlabel('Predicted')

    for i in range(2):
        for j in range(2):

            if i == 0 and j == 0:
                lab = 'TN: '
            elif i == 0 and j == 1:
                lab = 'FP: '
            elif i == 1 and j == 0:
                lab = 'FN: '
            else:
                lab = 'TP: '

            plt.text(j, i, lab+str(confusion[i, j]), ha='center', va='center', color='black', fontsize=15, bbox=dict(boxstyle="round", fc="w"))

    plt.title('Confusion Matrix')
    plt.show()


def plot_ROC_AUC(FPR, TPR, AUC):
    plt.figure()
    plt.plot(FPR, TPR, color='darkorange', lw=2, label=f'ROC curve (AUC = {AUC:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()



####################### VISUALIZATIONS #######################

def scatterplot_feats(feats_df: pd.core.frame.DataFrame, 
                      feat1: str, feat2: str, 
                      color_by: str, 
                      color_map = {1: 'red', 0: 'blue'}):

    '''
    Plot scatterplot of one feature against the other, colored by class (0, 1)
    '''
    plt.scatter(feats_df[feat1], feats_df[feat2], c=feats_df[color_by].map(color_map))

    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor=color_map[1], markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor=color_map[0], markersize=10)])

    plt.show()


def scatterplot_feats_3D(feats_df: pd.core.frame.DataFrame, 
                      feat_x: str, feat_y: str, feat_z: str, 
                      color_by: str, 
                      color_map = {1: 'red', 0: 'blue'}):
    '''
    Plot 3D scatterplot colored by class (0, 1),
    to show separation of classed when looking at 3 different features    
    '''
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for label_name, label_color in color_map.items():
        subset = feats_df[feats_df[color_by] == label_name]
        ax.scatter(subset[feat_x], subset[feat_y], subset[feat_z], c=label_color, label=label_name)

    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_zlabel(feat_z)

    ax.legend()

    plt.show()


def show_classification_results(res_test: pd.core.frame.DataFrame, rw: str,
                                seq = 'test_seq', true_lab = 'true_lab', pred_lab = 'pred_lab'):
    '''
    Plot either series correctly classified (if rw == 'right') or misclassified series (if rw == 'wrong')

    Input:
        -   res_test: pandas dataframe with results of model evaluation on test set; 
                    should contain test observations, their true label, and their predicted label
        -   rw: 'right' or 'wrong'
        -   seq, true_lab, pred_lab: columns names
    '''
    if rw == 'right':
        classif = res_test[res_test[true_lab] == res_test[pred_lab]]
        lab1 = 'Test Events'
        lab2 = 'Test Non-Events'
        title_plot = 'Right Classification'
    elif rw == 'wrong':
        classif = res_test[res_test[true_lab] != res_test[pred_lab]]
        lab1 = 'Test Events predicted as Non-Events'
        lab2 = 'Test Non-Events predicted as Events'
        title_plot = 'Classification Errors'
    else:
        raise ValueError('Invalid rw parameter!')

    zero_class = classif[classif[true_lab] == 0]
    one_class = classif[classif[true_lab] == 1]

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 

    for ev in one_class[seq].to_list():
        ax1.plot(ev)
    ax1.set_title(lab1)

    for non_ev in zero_class[seq].to_list():
        ax2.plot(non_ev)
    ax2.set_title(lab2)

    plt.suptitle(title_plot)

    plt.show()



