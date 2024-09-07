import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.metric import bi_classify_metrics, onehot, multi_classify_metrics, regression_metrics, plot_multiple_regressions,\
            bi_classify_metrics_names,multi_classify_metrics_names,regression_metrics_names

def read_scaffold_result(task_name='', model_name='', n_tasks=2):
    result_dir = f'pretrained/{task_name}/{model_name}/'
    result_files = os.listdir(result_dir)
    total_labels = []
    total_preds = []
    for result_file in result_files:
        if result_file.startswith('fold0') and result_file.endswith('txt'):
            result_path = result_dir+result_file
            labels = []
            preds = []
            with open(result_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    words = line.strip().split(', ')
                    labels.append(int(words[-1]) if n_tasks != 1 else float(words[-1]))
                    preds_words = ','.join(words[:-1])[1:-1].split(',')
                    temp_ls = []
                    for word in preds_words:
                        temp_ls.append(float(word))
                    preds.append(temp_ls if n_tasks!=1 else temp_ls[0])
            total_labels+=labels
            total_preds+=preds
    
    if n_tasks != 1:
        return np.array(onehot(np.array(total_labels), n_tasks)), np.array(total_preds)
    else:
        return np.array(total_labels),np.array(total_preds)

def plot_result(save_dir, classnames):
    df = pd.read_csv(f'{save_dir}/total_metric.txt', sep='\t', header=0, index_col=0)
    loc = 'lower right'
    if len(classnames) == 2:
        df = df.iloc[:, 4:]  
    elif len(classnames) == 1:
        df = df.drop(['MSE', 'SMAPE', 'EVAR', 'MAPE'], axis=1) 
        # loc = 'upper right'
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
    for column in df.columns:
        ax.plot(df.index, df[column], marker='o', label=column)
    ax.set_title('Performance Metrics for Models', fontsize=12)
    # ax.set_xlabel('Models', fontsize=10)
    ax.set_ylabel('Metric Value', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=7, loc=loc)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/total_metric.png')

def softmax(x, axis=None):
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def analysis(task_name, model_name, n_tasks=2, classnames=['non-Antibacterial','Antibacterial'], save_root_dir='pretrained'):
    save_dir = f'{save_root_dir}/{task_name}/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    labels, preds = read_scaffold_result(task_name, model_name, n_tasks)
    if n_tasks == 1:
        result = regression_metrics(labels, preds, plot_regress=True, save_path_name=f'{save_dir}/{model_name}',)
    elif n_tasks == 2:
        result = bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name=f'{save_dir}/{model_name}', classnames=classnames)
    else:
        result = multi_classify_metrics(labels, preds, average_method='micro', plot_cm=True, plot_auc=True, save_path_name=f'{save_dir}/{model_name}',classnames=classnames)
    np.savetxt(f'{save_dir}/kfolds_metric.txt', np.array(result))
    return labels, preds

def total_plot(task_name, model_names, classnames=['non-Antibacterial','Antibacterial'], save_root_dir='analysis/models_val_result'):
    save_dir = f'{save_root_dir}/{task_name}/'
    regress_labels, regress_preds = [], []
    total_metrics = []
    for model_name in model_names:        
        labels, preds = analysis(task_name, model_name, n_tasks=len(classnames), classnames=classnames, save_root_dir=save_root_dir)
        metric = np.loadtxt(save_dir+f'{model_name}/kfolds_metric.txt')
        total_metrics.append(metric)
        regress_labels.append(labels)
        regress_preds.append(preds)
    if len(classnames) == 1:
        header_names = regression_metrics_names
        plot_multiple_regressions(regress_labels, regress_preds, model_names, save_dir+'/total_regress.png')
    elif len(classnames) == 2:
        header_names = bi_classify_metrics_names
    else:
        header_names = multi_classify_metrics_names
    df = pd.DataFrame(total_metrics).round(2)
    df.columns = header_names
    df.index = model_names
    df.to_csv(save_dir+'/total_metric.txt', sep='\t')
    plot_result(save_dir, classnames)

if __name__ == '__main__':
    # total_plot('multitaste_mols', ['KPGT', 'MorganFP-MLP', 'MorganFP-RF', 'MorganFP-SVM', 'weaveGNN'], classnames=['Astringent','Bitter', 'Sweet', 'Salty', 'Sour', 'Kokumi', 'Tasteless']) # finished  
    # total_plot('antibacterial_mols', ['KPGT', 'MorganFP-MLP', 'MorganFP-RF', 'MorganFP-SVM', 'weaveGNN'], classnames=['non-Antibacterial','Antibacterial']) # finished
    # total_plot('astringent_mols', ['KPGT', 'MorganFP-MLP', 'MorganFP-RF', 'MorganFP-SVM', 'weaveGNN'], classnames=['Non-Astringent','Astringent']) # finished
    # total_plot('multitaste_peps', ['ESM-MLP', 'ESM-RF', 'ESM-SVM'], classnames=['Astringent','Bitter','Sweet','Salty','Sour','Umami','Kokumi']) # finished
    # total_plot('antibacterial_peps', ['ESM-MLP', 'ESM-RF', 'ESM-SVM'], classnames=['non-Antibacterial','Antibacterial']) # finished
    # total_plot('astringent_peps', ['ESM-MLP', 'ESM-RF', 'ESM-SVM'], classnames=['Non-Astringent','Astringent']) # finished
    # total_plot('astringent_mols_threshold', ['KPGT', 'MorganFP-MLP', 'MorganFP-RF', 'MorganFP-SVM', 'weaveGNN'], ['threshold'])  # finished
    # total_plot('astringent_peps_threshold', ['ESM-MLP', 'ESM-RF', 'ESM-SVM'], ['threshold'])  # finished
    
    # total_plot('afpeptide', ['ESM-MLP', 'ESM-RF', 'ESM-SVM'], classnames=['Non-AFP','AFP']) # finished
    # total_plot('afprotein', ['ESM-MLP', 'ESM-RF', 'ESM-SVM'], classnames=['Non-AFP','AFP']) # finished
    total_plot('tox_brain', ['MorganFP-MLP', 'weaveGNN'], classnames=['0','1'])
    total_plot('tox_pro', ['MorganFP-MLP', 'weaveGNN'], classnames=['0','1'])
    total_plot('tox_bone', ['MorganFP-MLP', 'weaveGNN'], classnames=['0','1'])