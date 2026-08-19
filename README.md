# record-breaking-llj-2026

This repository contains the scripts used in the research on an extreme Low-Level Jet (LLJ) event that occurred over South America between June 17 and 19, 2026.

**Scripts**

* `download_wyoming_data.py`: downloads sounding data from the University of [Wyoming website](https://weather.uwyo.edu/upperair/sounding.shtml). In addition to the required files, it requires the station number, start date, end date, and sounding times.

* `igra_data_process.py`: performs preprocessing of IGRAv2 (Integrated Global Radiosonde Archive) data. It only requires an [IGRAv2 file](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive).

* `llj_statistics.py`: performs statistical analysis, identifies extreme events, calculates LLJ persistence, and generates visualizations. In addition to the required files, it requires the city/station name, station file, coordinates, extreme-event percentile, and the number of soundings to be considered for persistence.

* `montini_criteria.py`: classifies Low-Level Jet (LLJ) events according to the methodology of [Montini et al. (2019)](https://doi.org/10.1029/2018JD029634). In addition to the required files, it requires the start and end dates. The remaining parameters follow the methodology described by Montini et al. (2019). If a different methodology is desired, the corresponding parameters must be modified.

* `process_wyoming_data.py`: preprocesses and standardizes atmospheric sounding data downloaded from the University of Wyoming into an IGRAv2-compatible format. It only requires the station number and the downloaded file.

* `skewt_satellite_images.py`: generates combined Skew-T diagrams, hodographs, and GOES satellite imagery for atmospheric analysis. In addition to the required files, it requires the sounding dates, satellite dates and information, and the coordinates of the stations, states, and countries.


**Citation**

Code DOI - If you use any of the scripts in this repository, please cite this software using the [DOI Repository](https://doi.org/10.5281/zenodo.21987558)

Dataset: The soundings data used in this research are available at [DOI Dataset](https://doi.org/10.5281/zenodo.21987615)

 Paper not published yet

**Authors**

M. S. Reboita, G. A. dos Santos, L. F. Gozzo, B. C. Capucin,  A. M. P. Nunes, C. P. G. Lopes, F. Vemado, M. S. Custódio, R. P. da Rocha

**License**

This repository is intended to support the reproducibility of the research presented in the associated publication.

