### Jazi_et.al_2023_noInt

This repository includes Python scripts we used to analyse and generate figures for M.N.Jazi et al., 2023 titled 'Hippocampal firing fields anchored to a moving object predict homing direction during path-integration-based behaviour'. 

### Instruction

The environment we use to run these notebooks is a single Python virtual environment equipped with three pre-installed packages. To begin using this environment, follow these steps to install the required packages in this order:

1. DeepLabcut https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/installation.md

2. SpikeA https://github.com/kevin-allen/spikeA

3. Autopipy https://github.com/kevin-allen/autopipy

Figures 1 and 2 contain a code to generate the first two main figures plus the first 4 Extended data figures. These figures are generated from only behavioural recordings from the ```autopi_behavior_2021``` dataset. ExtendedDataFig.5 is related to the control odour experiment and only includes behavioural data. The rest of the figures are generated from ```autopi_ca1```. Some extended data figures were generated in the notebook related to its main figure.

Please, before running any code, change ```allDataPath``` in ```setup_path.py``` to the path where you cloned the data directory.
Per default it is ```~/repo/Jazi_et.al_2023_noInt/data/Jazi_etal_2023_noInter```.


#### create virtual env for this project

```
mkdir ~/python_virtual_environments
python3 -m venv ~/python_virtual_environments/Jazi2023Env
source ~/python_virtual_environments/Jazi2023Env/bin/activate
```

#### install additional software

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
git clone https://github.com/kevin-allen/autopipy.git

pip install --upgrade pip

pip install -e ~/repo/spikeA
pip install -e ~/repo/autopipy
```
