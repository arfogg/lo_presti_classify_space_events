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


### Requirements

To install all needed dependencies, run the following command:

```bash
pip install -r requirements.txt
```