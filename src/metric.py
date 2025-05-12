
import os
from math import sqrt
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score,mean_absolute_error,mean_absolute_percentage_error

def onehot(labels,n_class):
    """print(onehot(np.array([0,1,2]),3))"""
    onehot = np.zeros((labels.shape[-1], n_class))
    for i, value in enumerate(labels):
        onehot[i, value] = 1
    return onehot

def de_onehot(labels):
    return np.argmax(labels, axis=1)

def Micro_OvR_AUC(labels, preds):
    fpr, tpr, _ = roc_curve(labels.ravel(), preds.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def Macro_OvR_AUC(labels, preds):
    n_classes = labels.shape[1]
    fpr={}
    tpr={}
    roc_auc={}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    return fpr_grid, mean_tpr, auc(fpr_grid, mean_tpr)

def Weighted_OvR_AUC(labels, preds, weighted_method="amount_divided"):
    n_classes = labels.shape[1]
    ls_num_equal1 = np.array([np.sum(labels[:,i] == 1) for i in range(n_classes)])
    if weighted_method=="equally_divided":
        reverse = 1/ls_num_equal1
        total = np.sum(reverse)
        weights = reverse / total
    else:
        weights = ls_num_equal1
    fpr={}
    tpr={}
    auc_list=[]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        auc_list.append(roc_auc_score(labels[:, i], preds[:, i]))
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    tpr_ls=[]
    for i in range(n_classes):
        tpr_ls.append(np.interp(fpr_grid, fpr[i], tpr[i]))  # linear interpolation
    weighted_mean_tpr = np.average(tpr_ls, axis=0, weights=weights)
    weighted_auc1 = auc(fpr_grid, weighted_mean_tpr)
    # weighted_auc2 = np.average(auc_list, weights=weights)
    # weighted_auc3 = roc_auc_score(labels, preds, multi_class="ovr", average="weighted",)                        
    return fpr_grid, weighted_mean_tpr, weighted_auc1

def plot_multiclassify_auc_curve(labels, preds, save_path="auc.png", classnames=['class1', 'class2', 'class3']):
    n_classes = labels.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], roc_auc['micro'] = Micro_OvR_AUC(labels, preds)
    fpr["macro"], tpr["macro"], roc_auc['macro'] = Macro_OvR_AUC(labels, preds)
    fpr["weighted"], tpr["weighted"], roc_auc['weighted'] = Weighted_OvR_AUC(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})", color="Gray", linestyle=":", linewidth=4)
    ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})", color="MediumOrchid", linestyle=":", linewidth=4)
    ax.plot(fpr["weighted"], tpr["weighted"], label=f"weighted-average ROC curve (AUC = {roc_auc['weighted']:.2f})", color="Peru", linestyle=":", linewidth=4)
    colors = cycle(['red', 'DeepSkyBlue', 'SeaGreen', 'MediumTurquoise', 'SteelBlue', 'LightPink', 'orange', 'LimeGreen'])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(labels[:, class_id], preds[:, class_id], name=f"ROC curve for {classnames[class_id]}", color=color, ax=ax)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC curves of One-vs-Rest multi-classification")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(prop={"size":8}, frameon=False)
    plt.tight_layout()
    fig.savefig(save_path)
    return roc_auc['micro'], roc_auc['macro'], roc_auc['weighted']

def plot_biclassify_auc_curve(y_true, y_pred, save_path="auc.png", classnames=['negative', 'positive']):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot(fpr, tpr, color='IndianRed', lw=2, label='Overall ROC curve (AUC = %0.2f)' % roc_auc)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(prop={"size":8}, frameon=False)
    plt.tight_layout()
    fig.savefig(save_path)
    return roc_auc

def plot_confuse_matrix(cm, classnames, save_path):
    cm = cm.astype(int)
    conf_matrix = pd.DataFrame(cm, index=classnames, columns=classnames)
    fig, ax = plt.subplots(figsize=(5,4), dpi=300)
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, cmap="Blues", ax=ax, fmt='d')
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    fig.savefig(save_path)

def plot_regress_curve(y_true, y_pred, pearson_corr, r2, rmse, save_path):
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    data = pd.DataFrame({'True':y_true, 'Predicted':y_pred})
    sns.regplot(x='True', y='Predicted', data=data, scatter_kws={'s': 50}, line_kws={'color': 'red', 'linewidth': 2})
    textstr = '\n'.join((
        f'$R^2={r2:.2f}$',
        f'$RMSE={rmse:.2f}$',
        f'$Pearson\\ Corr={pearson_corr:.2f}$'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('Regression Line with Performance Metrics', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path)


def plot_multiple_regressions(y_true_list, y_pred_list, names, save_path):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    colors = sns.color_palette("bright", len(y_true_list)) 
    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        r2 = r2_score(y_true, y_pred)
        data = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
        sns.regplot(x='True', y='Predicted', data=data, scatter_kws={'s': 50, 'color': colors[i]}, 
                    line_kws={'color': colors[i], 'linewidth': 2}, ax=ax)
        ax.text(0.05, 0.95 - i*0.05, f'{names[i]}: $R^2={r2:.2f}$', transform=ax.transAxes, 
                fontsize=12, color=colors[i], verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('Multiple Regression Lines with Performance Metrics', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path)

bi_classify_metrics_names = ['TN','FP','FN','TP','TPR/Recall/Sensitivity','TNR/Specificity','PRE','Accuracy','AP','F1-score','MCC','ROC-AUC']
def bi_classify_metrics(labels, preds, plot_cm=False, plot_auc=False, save_path_name='bi', classnames=['Negative','Positive']):
    y_true = de_onehot(labels)
    y_pred = de_onehot(preds)
    cm = confusion_matrix(y_true, y_pred) # default 1 as postive
    cm = cm.astype(np.float32)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0] 
    tp = cm[1][1]
    tpr = rec = sen = round(tp / (tp+fn+0.0001),3)
    tnr = spe = round(tn / (tn+fp+0.0001), 3)
    pre = round(tp / (tp+fp+0.0001), 3)
    acc = round((tp+tn) / (tp+fp+fn+tn+0.0001),3) # equal: accuracy_score(y_true, y_pred)
    f1 = round((2*pre*rec) / (pre+rec+0.0001),3) # equal: f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) # equal: mcc = (tp*tn-fp*fn) / ((tp+fp)*(fn+tp)*(fn+tn)*(fp+tn))**0.5
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    if plot_cm:
        save_path = save_path_name+'_cm.png'
        plot_confuse_matrix(cm, classnames, save_path)
    if plot_auc:
        save_path = save_path_name+'_auc.png'
        plot_biclassify_auc_curve(y_true, preds[:, 1], save_path, classnames)
    return np.array([tn,fp,fn,tp,tpr,tnr,pre,acc,ap,f1,mcc,auc])


multi_classify_metrics_names = ['Precision','Recall','Accuracy','F1-score','MCC','ROC-AUC']

def multi_classify_metrics(labels, preds, average_method='weighted',  plot_cm=False, plot_auc=False, save_path_name='multi', classnames=['class1','class2','class3','class4','class5']):
    y_true = de_onehot(labels)
    y_pred = de_onehot(preds)

    # plot cm
    cm = confusion_matrix(y_true, y_pred)
    tpr = sen = cm.diagonal() / cm.sum(axis=1)
    tnr = []
    for i in range(cm.shape[0]):
        mask = np.ones(cm.shape[0], dtype=bool)
        mask[i] = False
        tnr.append(np.sum(cm[mask][:, mask]) / np.sum(cm[mask]))
    tnr = spe = np.array(tnr)
    ## equal to below:
    # FP = cm.sum(axis=0) - np.diag(cm)  
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm )
    # TN = cm.sum() - (FP + FN + TP)
    # FP = FP.astype(float) 
    # FN = FN.astype(float)
    # TP = TP.astype(float)
    # TN = TN.astype(float)
    # TPR = TP/(TP+FN)
    # TNR = TN/(TN+FP)
    if plot_cm:
        save_path = save_path_name+'_cm.png'
        plot_confuse_matrix(cm, classnames, save_path)

    # plot roc and calculate auc
    if plot_auc:
        save_path = save_path_name+'_auc.png'
        micro_auc, macro_auc, weighted_auc = plot_multiclassify_auc_curve(labels, preds, save_path, classnames=classnames)
    else:
        if average_method == 'weighted':
            weighted_auc = roc_auc_score(labels, preds, multi_class="ovr", average="weighted",)   
        elif average_method == 'macro':
            macro_auc = roc_auc_score(labels, preds, multi_class="ovr", average="macro",)  
        else:
            micro_auc = roc_auc_score(labels, preds, multi_class="ovr", average="micro",)    
    if average_method == 'weighted':
        auc = weighted_auc
    elif average_method == 'macro':
        auc = macro_auc
    else:
        auc = micro_auc
    acc = accuracy_score(y_true, y_pred)

    # calculate pre,rec,acc,f1,mcc
    if average_method == 'weighted':
        pre = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    elif average_method == 'macro':
        pre = precision_score(y_true, y_pred, average='macro')
        rec =recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
    else:
        pre = precision_score(y_true, y_pred, average='micro')
        rec =recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
    mcc = matthews_corrcoef(y_true, y_pred)
    return np.array([pre,rec,acc,f1,mcc,auc])

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true+0.001) )) * 100

def calculate_smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

regression_metrics_names = ['Pearson correlation coefficient','R²','RMSE','MSE','EVAR','MAE','MAPE','SMAPE']

def regression_metrics(y_true, y_pred, plot_regress=True, save_path_name='regress'):
    pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred)) # mean_squared_error(y_true,y_pred,squared=False)
    evar = explained_variance_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    if plot_regress:
        save_path = save_path_name+'_regress.png'
        plot_regress_curve(y_true, y_pred, pearson_corr, r2, rmse, save_path)
    return np.array([pearson_corr,r2,rmse,mse,evar,mae,mape,smape])






