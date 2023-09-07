# this file defines where the data is saved
# download tarball from https://doi.org/10.5061/dryad.crjdfn39x
# and define the path it was extracted to

import os

allDataPath = "~/repo/Jazi_et.al_2023_noInt/data/Jazi_etal_2023_noInter"
allDataPath = os.path.expanduser(allDataPath) # needed if path is within home directory (starts with tilde in this case)
print("data path:", allDataPath)

if not os.path.isdir(allDataPath):
    raise IOError("path not found")

