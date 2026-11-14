import pandas as pd
import numpy as np
from helpers import get_event_label, generate_all_events, baseline_linear_model, get_performance
from sklearn.metrics import roc_curve, auc
from pprint import pprint

if __name__ == '__main__':
    tte_bins = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30]]
    #Load data
    data = pd.read_csv('./example_data.csv',index_col=0)
    data.index = pd.to_datetime(data.index)

    #Add in the linear fit baselines
    data['baseline_w5_h10'] =baseline_linear_model(data=data.SvO2,window=pd.Timedelta(5,'m'),
                                             horizon=pd.Timedelta(10,'m'))

    data['baseline_w10_h15'] = baseline_linear_model(data=data.SvO2, window=pd.Timedelta(10, 'm'),
                                                    horizon=pd.Timedelta(15, 'm'))

    above = data["ml_model"] >= 45
    data["alarm45"] = (above.rolling(window=3, min_periods=3).sum().eq(3)).astype(int)

    above = data["ml_model"] >= 75
    data["alarm75"] = (above.rolling(window=3, min_periods=3).sum().eq(3)).astype(int)

    below = data["baseline_w5_h10"] < 60
    data["baseline_w5_h10_alarm"] = (below.rolling(window=3, min_periods=3).sum().eq(3)).astype(int)

    below = data["baseline_w10_h15"] < 60
    data["baseline_w10_h15_alarm"] = (below.rolling(window=3, min_periods=3).sum().eq(3)).astype(int)

    for i in [62,64,66,68,70]:
        below = data["SvO2"] < i
        data["SvO2_{}alarm".format(i)] = (below.rolling(window=3, min_periods=3).sum().eq(3)).astype(int)

    #Get events
    data['hypoperfusion_flag'] = get_event_label(value=data['SvO2'].values, th=60, samples_per_minute=30)


    positive_events, negative_events = generate_all_events(df=data, label_name='hypoperfusion_flag')

    #Calculate time before for positive events
    positive_events['minutes_after_last_event'] = np.nan

    # Add additional flag that labels the 30 minutes prior to the event as 1 as well
    data['hypoperfusion_extended_flag'] = data['hypoperfusion_flag']
    for index,row in positive_events.iterrows():
        data.loc[row.Start - np.timedelta64(30,'m'):row.End, 'hypoperfusion_extended_flag'] = 1

    for i in range(1,len(positive_events)):
        time_diff = positive_events.Start.values[i] - positive_events.End.values[i-1]
        positive_events.loc[i,'minutes_after_last_event'] = time_diff / pd.Timedelta(minutes=1)


    #Taking inverse means the nan for the first event is taken
    positive_events = positive_events.loc[~(positive_events['minutes_after_last_event'] < 30)]
    positive_events = positive_events.loc[positive_events['min1_avg'] < 65]
    positive_events = positive_events.loc[positive_events['min5_avg'] < 70]

    data['time2event_mins'] = np.nan
    for index,row in positive_events.iterrows():
        event_data = data.loc[row.Start - np.timedelta64(30, 'm'):row.End].copy()
        data.loc[event_data.index.values,'time2event_mins'] = (pd.to_datetime(row.Start) - pd.to_datetime(event_data.index.values)) / np.timedelta64(1, 'm')

    #Generate sensitivity and specificity, by window
    alarm_models = ['alarm45','alarm75','baseline_w5_h10_alarm','baseline_w10_h15_alarm',
                    "SvO2_62alarm","SvO2_64alarm","SvO2_66alarm","SvO2_68alarm","SvO2_70alarm"]
    all_results = {}
    for am in alarm_models:

        all_results[am] = get_performance(df=data, alarm_col=am,
                            positive_events=positive_events, negative_events=negative_events,
                            time2event_bins=tte_bins)

        for te in tte_bins:
            bin_name = '{}-{}'.format(te[0], te[1])
            #Is null is negative data
            auc_data = data.loc[((data.time2event_mins >= te[0])&(data.time2event_mins <= te[1]))|(data.time2event_mins.isnull())].copy()
            fpr, tpr, thresholds = roc_curve(auc_data['hypoperfusion_extended_flag'], auc_data[am])
            roc_auc = auc(fpr, tpr)
            all_results[am][bin_name]['auroc'] = roc_auc


    pprint(all_results)


    #Generate PPV - per sample
    ppv_per_sample_res = {}
    for am in alarm_models:
        temp_ppv = data.loc[(data[am] == 1)&(data['hypoperfusion_extended_flag'] == 1)].shape[0]*100/data.loc[(data[am] == 1)].shape[0]
        ppv_per_sample_res[am] = temp_ppv

    # Generate PPV - per alarm event
    ppv_per_alarm_event_res = {}
    for am in alarm_models:
        positive_events, negative_events = generate_all_events(df=data, label_name=am)
        tp_count = 0
        total_count = 0
        for index,row in positive_events.iterrows():
            temp_data = data.loc[row.Start:row.End]
            if temp_data.hypoperfusion_extended_flag.sum() > 1:
                tp_count += 1
            total_count += 1
        ppv_per_alarm_event_res[am] = tp_count*100/total_count


    npv_per_sample_res = {}
    for am in alarm_models:
        temp_npv = data.loc[(data[am] == 0) & (data['hypoperfusion_extended_flag'] == 0)].shape[0] * 100 / \
                   data.loc[(data[am] == 0)].shape[0]
        npv_per_sample_res[am] = temp_npv

    print(ppv_per_sample_res)
    print(ppv_per_alarm_event_res)
    print(npv_per_sample_res)
    #Loop over each model, get alarm segments, calculate percent
    a = 1