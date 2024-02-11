# Detecting and classifying space weather events

At all times, the Earth is bombarded by charged particles emitted by the Sun. These particles carry their own magnetic field, which interacts with Earth's magnetosphere, leading to significant modifications in its behavior.  
This project studies such events, which can manifest as either Sudden Impulses (SI) or Sudden Storm Commencement (SSC), with the primary goal to automate the detection and classification of these phenomena.


### Data Sources

The data these project relies on are the following:

- [**Observatori de l'Ebre** data](https://www.obsebre.es/en/variations/rapid), which enable us to obtain the days between 2012 and 2017 in which a solar event took place, as well as its nature (SI or SSC). Such data are contained in the `sc_compact_1995_2019_d.csv` file.

- [**SuperMAG** data](https://supermag.jhuapl.edu/), reporting high-resolution (1 second) traces of the Earth's magnetic field, recorded from several magnetometers at different latitudes. We focused on data coming from event-days between 2012 and 2017, paying particular attention to the following information:
    - `Date_UTC`: timestamp
    - `IAGA`: name of the magnetometer
    - `MLT` and `MCOLAT`: information about magnetic local time and station magnetic co-latitude 
    - `dbz_nez`: the magnetometer traces along the North-component
Data from SuperMAG have been re-named following the convention `YYYY-MM-DD.csv`


### Project Structure

1. [`visualize_dBN.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/visualize_dBN.ipynb): studies and visualize the behaviour of the North-component of the Earth's magnetic field during an event; provides specifically the example of an event happened on 16th June 2012.
2. [`magnetometers_timeshift.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/magnetometers_timeshift.ipynb): produces an animation to visualize the time-shift in event detection across magnetometers at different latitudes.
3. [`classification_utils.py`](https://github.com/luisalopresti/classify_space_events/blob/main/classification_utils.py): useful functions for classification and visualization.
4. [`binary_windows_labelling.py`](https://github.com/luisalopresti/classify_space_events/blob/main/binary_windows_labelling.py): create labelled sequences, namely timeseries of 1201 seconds, labelled either as "event" or "non-event". An "event" is defined as a 20 minutes and 1 second window centered around the onset of a solar event, while a "non-event" represents a timeseries of 20 minutes and 1 second extracted from the portion of the day preceeding a solar event.

    **NOTE**: to run the script, set the data paths as follows:
    - *path_sc_compact*: path of the file recording all events from 1995 to 2019, obtained from the Observatori de l'Ebre
    - *events_dir*: directory containing all CSV of event-days at 1sec resolution, collected from SuperMAG
    - *event_path* & *non_event_path*: folders where to save event-labelled and non-event-labelled sequences, respectively

    The selected parameters (MLT, windows length, etc.) may also be easily changed previous to running the script.
5. [`3_classes_labelling.py`](https://github.com/luisalopresti/classify_space_events/blob/main/3_classes_labelling.py): replicate the process implemented in `binary_windows_labelling.py`, extending it to categorize event sequences as either SI or SSC.
6. [`explore_labelled_data.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/explore_labelled_data.ipynb): exploratory analysis of the labelled timeseries produces in the two previous scripts.
7. [`binary_classification_raw.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/binary_classification_raw.ipynb): binary classification (event vs. non-event) of raw labelled timeseries (without applying signal processing techniques).
8. [`binary_classification_transformed.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/binary_classification_transformed.ipynb): classification of the labelled timeseries, processing the signal through the Wavelet Scattering Network and applying traditional machine learning algorithms for classification.
9. [`binary_classification_with_feats_selection.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/binary_classification_with_feats_selection.ipynb): implement different feature selection strategies to reduce the number of scattering features needed for binary classification.
10. [`3_classes_task_raw.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/3_classes_task_raw.ipynb): multi-class classification (non-event vs. SI vs. SSC) on raw sequences.
11. [`3_classes_task_transformed.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/3_classes_task_transformed.ipynb): multi-class classification (non-event vs. SI vs. SSC), using the Wavelet Scattering Network as pre-processing strategy and traditional machine learning algorithms for classification.
12. [`3_classes_repeated_train_test_split.ipynb`](https://github.com/luisalopresti/classify_space_events/blob/main/3_classes_repeated_train_test_split.ipynb): reproduces multi-class classification for both RAW and TRANSFORMED sequences, over *k* different train-test splits; computes average performances over the different splits to assess result robustness.



### Requirements

To install all needed dependencies, run the following command:

```bash
pip install -r requirements.txt
```