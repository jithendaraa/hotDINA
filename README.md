# hotDINA

### Introduction
Fitting probabilistic student models like hotDINA_skill and hotDINA_full using PyMC3 and PyStan. Note: PyStan is faster for time series models like BKT/hotDINA etc.. This repo:

- Discusses how to set up the data for <a href="https://github.com/jithendaraa/RoboTutor-Analysis">RoboTutor-Analysis</a>

- Uses Michael Yudelson's tool <a href='https://github.com/myudelson/hmm-scalable'>hmmscalable</a> to fit BKT params to RoboTutor's transactions data. Fits and retrieves village wise params.

- Automatically fits the hotDINA_full and the hotDINA_skill model on a server (like PSC bridges) and retrieves the parameters for you using PyStan. <a href='https://www.cs.cmu.edu/~listen/pdfs/hoSM.pdf'>Link</a> to the hotDINA student model paper


### Installation and setup

1. In order to use/run this repo successfully, you will need to have the ```RoboTutor-Analysis``` repo cloned. The logged RoboTutor Data should be stored in ```RoboTutor-Analysis/Data/```. 
    - First clone the ```RoboTutor-Analysis``` project <br> ```git clone https://github.com/jithendaraa/RoboTutor-Analysis ``` <br>
    - Then, clone this project <br> ``` git clone https://github.com/jithendaraa/hotDINA ```.

2. Setting up transactions data and retrieving the BKT parameters: <br>
    Due to large amounts of logged data, the `Data` directory in `RoboTutor-Analysis` contains only the scripts to obtain the fitted BKT params and not the parameters themselves. <br>
    - Follow <a href='https://docs.google.com/document/d/1hcX1fhHzBLH3xweZrkdVJOty9yq4DAxCs_ZjNj1h9-c/edit'> this guide</a> to get village-specific transactions data (and the fitted BKT parameters for these 29 villages). At the end of this process, `RoboTutor-Analysis/Data` must contain 29 folders named `village_n` (n from 114 to 142), 1 hmmscalable folder, a `script.ipynb` and a `script.py`.

3. Setting up activity tables, CTA and other data in `RoboTutor-Analysis/Data` <br>
    1. Navigate to XPrize Home 2 folder and download the pristine form of the activity_table you need: Save this as `Activity_table_KCSubtest_sl2.xlsx` in the `Data` directory of `RoboTutor-Analysis`. **Note**: this activity table should have the last 4 columns as KC columns.
    2. Download `Code Drop 2 Matrices.xlsx` and `CTA.xlsx` into the `Data` directory

4. Look at the screenshot in the 3rd point <a href='https://github.com/jithendaraa/RoboTutor-Analysis#installation-and-setup'>here</a>. This is what your `Data` directory looks like after a successful setup!


### Server related setup (PSC Bridges)

The previous section already discussed about fitting BKT parameters using `hmmscalable`. This section focusses on using the scripts in `/scripts` to extract data from `RoboTutor-Analysis/Data`.

1. This example uses PSC bridges as the server but could be extended to any other supercomputing resource.

2. In this project, create a `passwords.py` with the following contents: <br>

```
PASSWORD = {
    "PSC":      "your_psc_password",
    "XSEDE":    "your_xsede_password"
}

PORT_NUM = xxxx (xxxx = the port number you want to use for ssh purposes)
SERVER = "xxxx.edu" (ssh username@xxxx -p PORT_NUM)
USERNAME = "xxxx" (ssh xxxx@SERVER -p PORT_NUM) 
```

## Usage 

### Extracting RT transactions data using scripts

1. `python get_data_for_village_n.py -v 130 -o 1200 `: Extracts transactions data of a single village (130) for the first 1200 attempts. Use `-o all` to get all attempts of a student in a particular village. Output: `pickles/data/data(village_num)_(num_obs).pickle`

2. `cd scripts && python get_data_for_villages.py -v 114-120 -o 1200`: Extracts transactions data for villages 114 to 120 for the first 1200 attempts. Use `-o all` to get all attempts of a student in a particular village. Output: `pickles/data/data(village_num)_(num_obs).pickle`.



