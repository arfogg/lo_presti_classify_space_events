import pandas as pd
import os
from datetime import datetime


'''
Create labelled data for binary classification problem.
Label data-windows as containing an event or a non-event.

NOTE: set the right data paths (search for the comment "PATHS"); 
if needed, may also change the selected parameters (search for "PARAMS").
'''


def check_NaN_proportion_of_given_station(single_iaga_df, seconds_in_each_window, NaN_threshold):
    '''
    Check if the proportion of NaN in the passed window is below threshold.

    Input:
        - single_iaga_df: dataframe containing timeseries for a single station/IAGA (pandas dataframe)
        - seconds_in_each_window: desired length of the above dataframe,
         namely the len when all timesteps at a second resolution are present in the dataframe (int)
        - NaN_threshold: (0., 1.); max percentage of NaN allowed (float)

    Return a tuple with values:
        - True if the percentage of NaN is below the threshold, False otherwise (bool)
        - float indicating the proportion of NaN in the window
    '''
    # compute proportion of NaN in window
    n_nan = single_iaga_df.dbn_nez.isna().sum() + (seconds_in_each_window - len(single_iaga_df))
    nan_proportion = n_nan/seconds_in_each_window
    if nan_proportion < NaN_threshold:
        # keep window
        return True, nan_proportion
    else:
        # drop window (too many NaN!)
        return False, nan_proportion
    

def fill_time_window(station_data, start, end):
    '''
    This function is used to fill missing timestep in the passed time series window.

    Input:
     - station_data: a dataframe with data for a single station/IAGA on a given day, indexes by datetime (pandas dataframe)
     - start, end: first and last timestamp of the window to be created (pandas timestamps)
     
    Returns:
     - filled_station_data: a dataframe, where missing timesteps (at 1 second resolution), 
     have been added (with NaN values for the corresponding variables). 
     The dataframe is sorted according to timestamps (indexes) 
    '''
    # fill missing steps around event
    date_range = pd.date_range(start=start, end=end, freq='1S') 
    missing_timestamps = date_range.difference(station_data.index)
    # add missing timestamps
    missing_data = pd.DataFrame(index=missing_timestamps)
    filled_window_data = pd.concat([station_data, missing_data])
    # return window with all timesteps
    return filled_window_data.sort_index()


def create_event_window(df, matching_timestamps, time_delta, seconds_in_each_window, NaN_threshold, event_path):
        '''
        Create an event-window for each event in the list matching_timestamps.
        Each window has length time_delta*2 minutes and is centered around the event.
        Only time-windows that have a proportion of NaN below the NaN_threshold are stored under the event_path directory.

        Input:
            - df: pandas dataframe containing one day of data for all available stations/IAGA
            - matching_timestamps: list of timestamps containing all the solar events of the day
            - time_delta: minutes to be considered before and after the event
            - seconds_in_each_window: length of the single-station-timeseries when all timesteps are present
            - NaN_threshold: max proportion of NaN to be allowed in a timeseries
            - event_path: folder where to save the created event-windows

        Output:
            - save event-window in the event_path folder, using name station_timestamp_event
        
        '''
        for ts in matching_timestamps: # usually a single timestamp, at times 2 or 3 events per day
            # create window centered around ts event, for a single station
            start, end = ts - pd.Timedelta(minutes=time_delta), ts + pd.Timedelta(minutes=time_delta)

            # filter the dataframe according to given timeframe
            filtered_df = df[(df.index <= end) & (df.index >= start)]

            # loop over all available stations for the given timeframe
            for iaga in filtered_df.IAGA.unique():
                # take dataframe for a single stations
                single_iaga = filtered_df[filtered_df.IAGA == iaga]

                # check if proportion of NaN is below the acceptable threshold
                if check_NaN_proportion_of_given_station(single_iaga, seconds_in_each_window, NaN_threshold)[0]:
                    filled_single_iaga = fill_time_window(single_iaga, start, end)
                    # save
                    filled_single_iaga.to_csv(event_path+iaga+'_'+str(ts).replace(' ', '_')+'.csv')


def create_non_event_window(df, ts, time_delta, seconds_in_each_window, NaN_threshold, non_event_path):
    '''
    This function extracts for each event days, the time series going from the beginning of the day until the onset of the (first) event,
    and recursively split it in time_delta*2 minutes windows,
    going from the event's onset backwards.
    NOTE: we discarded the part of the time series following the (first) event due to possible long-lasting effects (hours) of events.

    Non-event windows are stored in the non_event_path folder 
    if the percentage of NaN in the window does not exceed the passed threshold.

    The produced non-event-windows are of the same length of the event ones
    (event-window len = 10 min before + 10 min after the event + 1 sec of the event)

    Input:
        - df: pandas dataframe containing one day of data for all available stations/IAGA
        - ts: timestamp of the first solar event of the day 
        - time_delta: minutes to be considered before and after the event
        - seconds_in_each_window: length of the single-station-timeseries when all timesteps are present
        - NaN_threshold: max proportion of NaN to be allowed in a timeseries
        - non_event_path: folder where to save the created event-windows
    
    Output:
        - save non-event-window in the non_event_path folder, using name station_timestamp_start_window
    '''

    start_event_window = ts - pd.Timedelta(minutes=time_delta)
    df_before_event = df[(df.index <= start_event_window)] # 1sec overlap bc of =

    # CREATE NON-EVENT WINDOWS BEFORE TS
    # from TS going BACKWARDS until the start of the day
    ts_end_window = start_event_window
    ts_start_window = ts_end_window - pd.Timedelta(minutes=time_delta*2)
    while ts_start_window.date() == ts.date(): 
        # while start of non-event window is in the day of event
        # (this means that if going back in time the last window of the day 
        # is less than time_delta*2 minutes in length, it is going to be ignored)
        window_df = df_before_event[(df_before_event.index <= ts_end_window) & (df_before_event.index >= ts_start_window)]

        for iaga in window_df.IAGA.unique():
            single_iaga = window_df[window_df.IAGA == iaga]
            if check_NaN_proportion_of_given_station(single_iaga, seconds_in_each_window, NaN_threshold)[0]:
                filled_single_iaga = fill_time_window(single_iaga, ts_start_window, ts_end_window) 
                # save
                filled_single_iaga.to_csv(non_event_path+iaga+'_'+str(ts_start_window).replace(' ', '_')+'.csv')

        # update window extremes
        ts_end_window = ts_start_window
        ts_start_window = ts_end_window - pd.Timedelta(minutes=time_delta*2)








def main():
    # PARAMS
    time_delta = 10 # minutes
    NaN_threshold = .3 # max percentage NaN allowed
    seconds_in_each_window = time_delta*60*2 + 1 # seconds in each window (window_df length)
    start_year, end_year = 2012, 2017 # years-range of the events considered
     
    MLT = (11.30, 12.30) # MAGNETIC LOCAL TIME
    tollerance = 0.

    create_event, create_non_event = True, True

    # PATHS
    path_sc_compact = 'data/sc_compact_1995_2019_d.csv' # path of file recording all events from 1995 to 2019, from Observatori de l'Ebre
    events_dir = 'data/events/' # directory containing all events csv files at 1sec resolution from SuperMAG
    events = os.listdir(events_dir) # list of csv of events
    events.sort()
    event_path = 'labelled_data/event/' # folder where to SAVE event-labelled windows
    non_event_path = 'labelled_data/non-events/' # folder where to SAVE non-event-labelled windows

    ###########################

    # ALL EVENTS LIST

    # get list of timestamps of all events from sc_compact_1995_2019_d dataset 
    all_events = pd.read_csv(path_sc_compact)
    # add timestamps to dataframe
    all_events['Date_UTC'] = pd.to_datetime(all_events[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1)+' '+all_events[['hour', 'minute']].astype(str).agg(':'.join, axis=1), format='%Y-%m-%d %H:%M.%S')
    # timestamps of all events
    event_timestamps = all_events.Date_UTC.to_list()

    # filter events per year (get events in the considered year-range)
    event_timestamps = [ts for ts in event_timestamps if start_year <= ts.year <= end_year]


    ###########################

    # FOR EACH EVENT-DAY, CREATE AND SAVE THE EVENT WINDOW 
    # (time_delta*2 minutes window, centered around the event)

    for e in events:
        df = pd.read_csv(events_dir+e)
        df = df[(df.MLT >= MLT[0] - tollerance) &(df.MLT <= MLT[1] + tollerance)].reset_index(drop=True) 
        if len(df) > 0:
            df['Date_UTC'] = pd.to_datetime(df.Date_UTC)
            df.set_index('Date_UTC', inplace=True)

            # get timestamps of the event(s) on the target day only
            target_day = df.index.date[0]

            matching_timestamps = []

            for timestamp in event_timestamps:
                if timestamp.date() == target_day:
                    matching_timestamps.append(timestamp)

            # save laballed data

            if create_event:
                # create and save event windows
                create_event_window(df, matching_timestamps, time_delta, seconds_in_each_window, NaN_threshold, event_path)
            
            if create_non_event:
                # create and save non-event windows
                create_non_event_window(df, min(matching_timestamps), time_delta, seconds_in_each_window, NaN_threshold, non_event_path)


if __name__ == "__main__":
    main()







