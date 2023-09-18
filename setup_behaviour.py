import pandas as pd
import numpy as np
import os.path
import shutil
from autopipy.project import Project

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

import pandas as pd
import numpy as np
import os.path
import importlib
import autopipy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from autopipy.project import Project
importlib.reload(autopipy.project)
from autopipy.project import Project

from autopipy.trial import Trial
importlib.reload(autopipy.trial)
from autopipy.trial import Trial

from autopipy.session import Session
importlib.reload(autopipy.session)
from autopipy.session import Session
from scipy.stats import wilcoxon,ttest_rel


from setup_path import *

projectName="autopi_behavior_2021"
dataPath = allDataPath + "/" + projectName # this is for behaviour

myProject = Project(name=projectName,dataPath=dataPath)
