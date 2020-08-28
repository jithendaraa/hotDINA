# hotDINA

### Introduction
Fitting probabilistic student models like hotDINA_skill and hotDINA_full using PyMC3 and PyStan. Note: PyStan is faster for time series models like BKT/hotDINA etc.. This repo:

- Discusses how to set up the data for <a href="https://github.com/jithendaraa/RoboTutor-Analysis">RoboTutor-Analysis</a>

- Uses Michael Yudelson's tool <a href='https://github.com/myudelson/hmm-scalable'>hmmscalable</a> to fit BKT params to RoboTutor's transactions data. Fits and retrieves village wise params.

- Automatically fits the hotDINA_full and the hotDINA_skill model on a server (like PSC bridges) and retrieves the parameters for you using PyStan. <a href='https://www.cs.cmu.edu/~listen/pdfs/hoSM.pdf'>Link</a> to the hotDINA student model paper


### Instructions and Usage

1. In order to use/run this repo successfully, you will need to have the ```RoboTutor-Analysis``` repo cloned. The logged RoboTutor Data should be stored in ```RoboTutor-Analysis/Data/```. 
    - First clone the ```RoboTutor-Analysis``` project <br> ```git clone https://github.com/jithendaraa/RoboTutor-Analysis ``` <br>
    - Then, clone this project <br> ``` git clone https://github.com/jithendaraa/hotDINA ```.

2. Setting up transactions data and retrieving the BKT parameters: <br>
    Due to large amounts of logged data, the `Data` directory in `RoboTutor-Analysis` contains only the scripts to obtain the fitted BKT params and not the parameters themselves. <br>
    - Follow <a href='https://docs.google.com/document/d/1hcX1fhHzBLH3xweZrkdVJOty9yq4DAxCs_ZjNj1h9-c/edit'> this guide</a> to get village-specific transactions data (and the fitted BKT parameters for these 29 villages). At the end of this process, `RoboTutor-Analysis/Data` must contain 29 folders named `village_n` (n from 114 to 142), 1 hmmscalable folder, a `script.ipynb` and a `script.py`.

3. Setting up activity tables, CTA and other data in `RoboTutor-Analysis/Data` <br>
    1. Navigate to XPrize Home 2 folder and download the pristine form of the activity_table you need: Save this as `Activity_table_KCSubtest_sl2.xlsx` in the `Data` directory of `RoboTutor-Analysis`. Note: this activity table should have the last 4 columns as KC columns.
    2. Download `Code Drop 2 Matrices.xlsx` and `CTA.xlsx` into the `Data` directory



