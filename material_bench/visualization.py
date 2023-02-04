import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.font_manager as font_manager
import math

def plot_best_trend(perfomance_dataset_dic,baseline_dataset=None, plot_pred = True):

    fig = plt.figure(figsize=(12,12))
    ax0 = fig.add_subplot(111)

    if baseline_dataset is not None:
        n_b_data = len(baseline_dataset.data)
        best_value = np.min(baseline_dataset.data[baseline_dataset.y_name_list].values)
        if baseline_dataset.direction_adjustment == True:
            best_value = -1 * best_value
        else:
            pass

        ax0.plot(np.linspace(1, n_b_data, n_b_data), np.full(n_b_data,best_value),'--',color='black',linewidth = 3)    

    color_list = ["r", "b", "g", "y", "m", "c"]
    i = 0
    for perfomance_k, perfomance_i in perfomance_dataset_dic.items():
        n_p_data = len(perfomance_i['y_best_trend'])
        y_trend = perfomance_i['y_best_trend']
        if baseline_dataset.direction_adjustment == True:
            y_trend = -1 * y_trend     
        else:
            pass

        font = font_manager.FontProperties(family='Arial', size = 26, style='normal')
        ax0.plot(np.arange(n_p_data) + 1, y_trend, label = perfomance_k, color = color_list[i], linewidth=3)
        ax0.legend(prop = font)
        ax0.set_xlim([0, 300])
        i += 1

    if plot_pred is True:
        for perfomance_k, perfomance_i in perfomance_dataset_dic.items():
            n_p_data = len(perfomance_i['y_pred_mean'])
            y_pred = perfomance_i['y_pred_mean']
            if baseline_dataset.direction_adjustment == True:
                y_pred = -1 * y_pred     
            else:
                pass
            ax0.plot(np.arange(n_p_data) + 1, y_pred,'--', label = perfomance_k, color = color_list[i], linewidth=3)    
            y_low = y_pred - 1.96 * perfomance_i['y_pred_std']
            y_high = y_pred + 1.96 * perfomance_i['y_pred_std']
            ax0.fill_between(np.arange(n_p_data) + 1, y_low,y_high, color = color_list[i], alpha=0.2)

        
    #ax0.set_xscale('log')

def plot_same_ac_value_trend(perfomance_dataset_dic,plot_type = 'same_value_pred_count'):

    fig = plt.figure(figsize=(12,12))
    ax0 = fig.add_subplot(111)

    color_list = ["r", "b", "g", "y", "m", "c"]
    i = 0
    for perfomance_k, perfomance_i in perfomance_dataset_dic.items():
        n_p_data = len(perfomance_i[plot_type])
        ax0.plot(np.arange(n_p_data) + 1, perfomance_i[plot_type], label = perfomance_k, color = color_list[i], linewidth=3)
        
        i += 1
    
    font = font_manager.FontProperties(family='Arial', size = 26, style='normal')
    ax0.legend(prop = font)
    ax0.set_xlim([0, 300])