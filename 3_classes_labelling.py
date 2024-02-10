from binary_windows_labelling import fill_time_window, check_NaN_proportion_of_given_station
import os
import pandas as pd


'''
Create labelled data for multiclass problem.
Label data-window as non-event, SI or SSC (3 classes).
'''

def create_SI_SSC_window(df, matching_timestamps, type_event_dict, time_delta, seconds_in_each_window, NaN_threshold, event_path):
        '''
        Create an event-window for each event in the list matching_timestamps, and label the window as SC or SSC.
        Each window has length time_delta*2 minutes and it centered around the event.
        Only time-windows that have a proportion of NaN below the NaN_threshold are stored under the event_path directory.

        Input:
            - df: pandas dataframe containing one day of data for all available stations/IAGA
            - matching_timestamps: list of timestamps containing all the solar events of the day
            - type_event_dict: dict with type of event corresponding to each timestamp (timestamp as key, event type as value - SI or SSC)
            - time_delta: minutes to be considered before and after the event
            - seconds_in_each_window: length of the single-station-timeseries when all timesteps are present
            - NaN_threshold: max proportion of NaN to be allowed in a timeseries
            - event_path: folder where to save the created event-windows

        Output:
            - save event-window in the event_path folder, under a nested repository (either SI or SSC), 
             with the name station_timestamp_event
        
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
                    filled_single_iaga.to_csv(event_path+type_event_dict[ts]+'/'+iaga+'_'+str(ts).replace(' ', '_')+'.csv')


def create_window_before_event_ts(df, ts, time_delta, seconds_in_each_window, NaN_threshold, non_event_path):
    '''
    This function extracts for each event days, the time series going from the beginning of the day until the onset of the (first) event,
    and recursively split it in time_delta*2 minutes windows,
    going from the event's onset backwards.
    NOTE: we discarded the part of the time series following the (first) event due to possible long-lasting effects (hours) of events.
    
    Non-event windows are stored in the non_event_path folder 
    if the percentage of NaN in the window does not exceed the passed threshold.

    The produced non-event-windows are of the same length of event ones
    (event-window len = 10 min before + 10 min after the event + 1 sec of the event)

    Input:
        - df: pandas dataframe containing one day of data for all available stations/IAGA
        - ts: timestamp of the first solar event of the day 
        - time_delta: minutes to be considered before the event
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
    time_delta = 10
    NaN_threshold = .3
    seconds_in_each_window = time_delta*60*2 + 1 # seconds in each window (window_df length)
    start_year, end_year = 2012, 2017 # years-range of the events considered
     
    MLT = (11.30, 12.30) # MAGNETIC LOCAL TIME
    tollerance = 0.

    create_event, create_non_event = False, True # do not need to create non-event, they're the same of the binary problem

    # PATHS
    path_sc_compact = 'data/sc_compact_1995_2019_d.csv' # path of file recording all events times (sc_compact_1995_2019_d.csv)
    events_dir = 'data/events/' # directory containing all events csv files at 1sec resolution from SuperMAG
    events = os.listdir(events_dir) # list of csv of events
    events.sort()
    event_path = 'labelled_data/' # folder where to save event-labelled windows
    non_event_path = 'labelled_data/before_event/' # folder where to save non-event-labelled windows (normal situation previous to event)


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

    # get the type of event corresponding to each timestamp (i.e., SI or SSC)
    type_event = all_events[all_events.Date_UTC.isin(event_timestamps)][['Date_UTC', 'type']]
    type_event_dict = type_event.set_index('Date_UTC')['type'].to_dict()


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
                create_SI_SSC_window(df, matching_timestamps, type_event_dict, time_delta, seconds_in_each_window, NaN_threshold, event_path)
            
            if create_non_event:
                # create and save non-event windows
                create_window_before_event_ts(df, min(matching_timestamps), time_delta, seconds_in_each_window, NaN_threshold, non_event_path)



if __name__ == "__main__":
    main()