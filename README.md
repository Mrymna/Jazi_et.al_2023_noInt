# Jazi_et.al_2023_noInt


This repository includes the Python scripts and notebooks used to analyze data and generate figures for the manuscript from M.N.Jazi and colleagues, 2023, titled 'Hippocampal firing fields anchored to a moving object predict homing direction during path-integration-based behaviour.'


The analysis was performed on a computer running Ubuntu 20.04.


## Clone this repository


```
cd ~
mkdir repo
cd repo
https://github.com/Mrymna/Jazi_et.al_2023_noInt.git
```


## Data set


You will need to download the data set from the Dryad repository. You can use this link to find the dataset: https://doi.org/10.5061/dryad.crjdfn39x


The data set is split into two files because of the large size of the data set. You can combine them in a Linux terminal using the following command.


 ```
cat Jazi_etal_2023_noInter_part1.tar.gz Jazi_etal_2023_noInter_part2.tar.gz > Jazi_etal_2023_noInter.tar.gz
```


## Create your Python environment


The commands below allow you to set up a Python environment in which you should be able to run the scripts and notebooks provided in this repository. 


Create your environment.


```
mkdir ~/python_virtual_environments
python3 -m venv ~/python_virtual_environments/Jazi2023Env
```


Activate your environment and install the packages listed in the `requirements.txt`.


```
source ~/python_virtual_environments/Jazi2023Env/bin/activate
cd ~/repo/Jazi_et.al_2023_noInt
python -m pip install -r requirements.txt
```


Make venv available for jupyter notebook.


```
pip install --user ipykernel
python -m ipykernel install --user --name=Jazi2023Env
```
Run Jupyter Lab or Jupyter Notebook.


```
jupyter lab
# jupyter notebook
```

## Install the spikeA and autopipy python packages


spikeA: https://github.com/kevin-allen/spikeA
autopipy: https://github.com/kevin-allen/autopipy


Make sure your environment is activated and install the two packages.

```
pip install --upgrade pip
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
cd ~/repo/spikeA
pip install -r requirements.txt
pip install -e ~/repo/spikeA
cd ~/repo/spikeA/spikeA/
python setup.py build_ext --inplace
```


```
cd ~/repo
git clone https://github.com/kevin-allen/autopipy.git
git checkout 91e6bf671731119f067a65ce7a97f00bbd427ed0
pip install -e ~/repo/autopipy
```

# File description 

`/fig1/Figure1.ipynb` and `/fig2/Figure2.ipynb` contain the code to generate the first two main figures plus the first 4 Extended data figures. These figures are generated from behavioral data located in the ```autopi_behavior_2021``` dataset. `/ExtDataFig5/Olfac_perMouse.ipynb` is related to the control odour experiment and only includes behavioural data. The rest of the figures are generated from ```autopi_ca1```. Some extended data figures were generated in the notebook related to its main figure.


Please, before running any code, change ```allDataPath``` in ```setup_path.py``` to the path where you cloned the data directory. Per default it is ```~/repo/Jazi_et.al_2023_noInt/data/Jazi_etal_2023_noInter```.
