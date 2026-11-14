import numpy as np
import pandas as pd

def get_event_label(value,th,samples_per_minute):
    base_values = value.copy()
    base_values[value <= th] = 1
    base_values[value > th] = 0
    diff = np.diff(base_values)
    diff[diff < 0] = 0
    diff = diff[~np.isnan(diff)]
    event_starts = np.where(diff == 1)[0]
    for es in event_starts:
        if (np.sum(base_values[es+1:es+1+samples_per_minute]) != samples_per_minute):
            base_values[es+1:es+1+samples_per_minute] = 0
    if value[0] == 1:
        if (np.sum(base_values[:samples_per_minute]) != samples_per_minute):
            base_values[:samples_per_minute] = 0
    return base_values

def baseline_linear_model(data,window,horizon):

    idx = pd.DatetimeIndex(data.index)

    window_td = pd.Timedelta(window)
    horizon_td = pd.Timedelta(horizon)

    # Convert timestamps to seconds from first sample for stable regression math.
    t_ns = idx.to_numpy(dtype="datetime64[ns]").astype("int64")
    t_sec = (t_ns - t_ns[0]) / 1e9
    y = data.astype(float).to_numpy()

    out = -1*np.ones(len(data))
    window_ns = int(window_td.value)
    horizon_ns = int(horizon_td.value)

    for i in range(len(data)):
        now_ns = t_ns[i]
        start_ns = now_ns - window_ns

        left = np.searchsorted(t_ns, start_ns, side="left")
        right = i + 1

        xw = t_sec[left:right]
        yw = y[left:right]

        valid = ~np.isnan(yw)
        xw = xw[valid]
        yw = yw[valid]

        if len(yw) < 2:
            continue

        xm = xw.mean()
        ym = yw.mean()
        denom = np.sum((xw - xm) ** 2)
        if denom == 0:
            continue

        m = np.sum((xw - xm) * (yw - ym)) / denom
        b = ym - m * xm

        now_sec = t_sec[i]
        end_sec = (now_ns + horizon_ns - t_ns[0]) / 1e9
        y_now = m * now_sec + b
        y_end = m * end_sec + b
        out[i] = min(y_now, y_end)

    return pd.Series(out, index=idx)


def get_events(data, label):
    # Get the label, pad it, and then difference it
    label_temp = data[label].values
    label_temp = np.pad(label_temp, 1, mode='constant', constant_values=0)
    label_temp_dif = np.diff(label_temp, axis=0)

    # Get the start and ends using the differenced label
    starts = data.index.values[np.where(label_temp_dif == 1)[0]]
    ends = data.index.values[np.where(label_temp_dif == -1)[0] - 1]

    # put into dataframe
    se_df = pd.DataFrame(data={'Start': starts, 'End': ends}, columns=['Start', 'End'])

    # Grab some extra details about each event
    se_df['Duration'] = (se_df['End'] - se_df['Start']) / np.timedelta64(1, 'm')
    se_df['min1_avg'] = np.nan
    se_df['min5_avg'] = np.nan

    # For each event grab specific information like time weighted average, difference in svo2 at start of event vs
    # right before
    for index, row in se_df.iterrows():

        # Set backwards label before to 1
        min1_avg = data.loc[row.Start - np.timedelta64(90, 's'):row.Start - np.timedelta64(30, 's'), 'SvO2'].mean()
        min5_avg = data.loc[row.Start - np.timedelta64(330, 's'):row.Start - np.timedelta64(270, 's'), 'SvO2'].mean()

        se_df.loc[index, 'min1_avg'] = min1_avg
        se_df.loc[index, 'min5_avg'] = min5_avg

    return data, se_df


def generate_all_events(df, label_name):
    # Referred to global hypoperfusion progression window in docs
    event_buffer = np.timedelta64(30, 'm')

    # load data
    data = df.loc[df.SvO2.notnull()].copy()

    data.index = data.index.astype('datetime64[ns]')

    # Get Positive events
    data, positive_events = get_events(data=data, label=label_name)

    data[label_name + '_expanded'] = 0

    # Run through each valid backwards event
    for index, row in positive_events.iterrows():
        data.loc[row.Start - event_buffer:row.End, label_name + '_expanded'] = 1

    data['negative_event_label'] = 1
    data.loc[data[label_name + '_expanded'] == 1, 'negative_event_label'] = 0
    data.loc[data['SvO2'] <= 60, 'negative_event_label'] = 0

    data, negative_events = get_events(data=data, label='negative_event_label')

    return positive_events, negative_events

def get_performance(df,alarm_col,positive_events,negative_events,time2event_bins):

    performance = {'tte_bin':[],'type':[]}
    for index,row in positive_events.iterrows():
        event_data = df.loc[row.Start - np.timedelta64(30,'m'):row.Start].copy()
        for t2e_bin in time2event_bins:
            assessment_data = event_data.loc[(event_data.time2event_mins <= t2e_bin[1])&(event_data.time2event_mins >= t2e_bin[0])]

            performance['tte_bin'].append('{}-{}'.format(t2e_bin[0],t2e_bin[1]))
            if assessment_data[alarm_col].sum() > 0:
                performance['type'].append('TP')
            else:
                performance['type'].append('FN')

    for index,row in negative_events.iterrows():
        event_data = df.loc[row.Start:row.End].copy()
        window_start = row.Start
        window_end = row.Start + np.timedelta64(5,'m')
        while window_end < row.End:
            assessment_data = event_data.loc[window_start:window_end]

            performance['tte_bin'].append('NA')
            if assessment_data[alarm_col].sum() > 0:
                performance['type'].append('FP')
            else:
                performance['type'].append('TN')

            window_start += np.timedelta64(5,'m')
            window_end += np.timedelta64(5,'m')

    performance = pd.DataFrame(performance)

    results = {}

    for t2e_bin in time2event_bins:
        bin_name = '{}-{}'.format(t2e_bin[0], t2e_bin[1])
        temp_performance = performance.loc[performance.tte_bin.isin(['NA', bin_name])]
        sensitivity = temp_performance.loc[temp_performance.type == 'TP'].shape[0] * 100 / (
                    temp_performance.loc[temp_performance.type == 'TP'].shape[0] +
                    temp_performance.loc[temp_performance.type == 'FN'].shape[0])
        specificity = temp_performance.loc[temp_performance.type == 'TN'].shape[0] * 100 / (
                    temp_performance.loc[temp_performance.type == 'TN'].shape[0] +
                    temp_performance.loc[temp_performance.type == 'FP'].shape[0])
        results[bin_name] = {}
        results[bin_name]['sensitivity'] = sensitivity
        results[bin_name]['specificity'] = specificity
    return results
