### Jazi_et.al_2023_noInt

This repository includes the Python scripts and notebooks that were used to analyze data and generate figures for the manuscript from M.N.Jazi and colleagues, 2023, titled 'Hippocampal firing fields anchored to a moving object predict homing direction during path-integration-based behaviour'.

### Instruction

To run the code, you will need to set a python environment with several packages.  To begin setting up this environment, follow the steps to install the required packages in this order:

1. SpikeA https://github.com/kevin-allen/spikeA

2. Autopipy https://github.com/kevin-allen/autopipy

3. Download the database from datadryad with this address: https://doi.org/10.5061/dryad.crjdfn39x

4. Merge the two tar files downloaded from datadryad using this command:  ```cat Jazi_etal_2023_noInter_part1.tar.gz Jazi_etal_2023_noInter_part2.tar.gz > Jazi_etal_2023_noInter.tar.gz```

Figures 1 and 2 contain the code to generate the first two main figures plus the first 4 Extended data figures. These figures are generated from behavioral data located in the ```autopi_behavior_2021``` dataset. ExtendedDataFig.5 is related to the control odour experiment and only includes behavioural data. The rest of the figures are generated from autopi_ca1. Some extended data figures were generated in the notebook related to its main figure.

Please, before running any code, change ```allDataPath``` in ```setup_path.py``` to the path where you cloned the data directory. Per default it is ```~/repo/Jazi_et.al_2023_noInt/data/Jazi_etal_2023_noInter```.


#### create virtual env for this project

```
mkdir ~/python_virtual_environments
python3 -m venv ~/python_virtual_environments/Jazi2023Env
```

to activate it and install requirements
```
source ~/python_virtual_environments/Jazi2023Env/bin/activate
python -m pip install -r requirements.txt
```

make venv available for jupyter notebook
```
pip install --user ipykernel
python -m ipykernel install --user --name=Jazi2023Env
```

run jupyter lab or notebook
```
jupyter lab
# jupyter notebook
```

#### install additional software

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
git clone https://github.com/kevin-allen/autopipy.git

pip install --upgrade pip

pip install -e ~/repo/spikeA
pip install -e ~/repo/autopipy

# in order to work with the pickle files, you need this version of autopipy, run that command in the ~/repo/autopipy directory (where you cloned the repo)
git checkout 91e6bf671731119f067a65ce7a97f00bbd427ed0 .

```
