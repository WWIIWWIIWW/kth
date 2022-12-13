# kth
------------
``kth`` contains some packages we build at KTH, Royal Institute of Technology, Sweden for multiple purpose tasks. We will constantly update sub-packages contained in it.

1. cantera collections for flame calculations.

2. keyfi collections for dimensional reduction task and combustion visualization - also see [keyfi](https://github.com/marrov/keyfi)

3. NN collections for building all kinds of neural network.
## Author
------------
[Kai Zhang, KTH, Royal Institute of Technology, Sweden](https://scholar.google.com/citations?user=lfUyemMAAAAJ&hl=en) - Google Scholar

Email: kaizhang@kth.se; kai.zhang.1@city.ac.uk;

### Installation
------------
These instructions will get you a copy of the project up and running on your local machine for usage, development and testing purposes. Please note that only Linux environments have been tested in the current implementation but the code should work independently of the OS.

Open a terminal window and clone this repository by writing:
```bash
git clone https://github.com/WWIIWWIIWW/kth
```
In order to use ``kth`` several Python 3 packages are required. Creating a brand new Conda environment for is recommended. This can be done easily with the provided environment.yml file as follows:
```bash
conda env create -f environment.yml
conda activate kth
```
After executing these commands a new Conda environment named ``kth`` will be created with all necessary packages. The environment is self-contained so as to not influence other local python installations and avoid conflicts with previously installed packages.

Now you can install the ``kth`` package into this newly created environment that contains all requirements by moving into the main folder and using the pip install command as follows:
```bash
pip install -e .
```
You can test out your installation by running the ``main.py`` in the **tutorial** directory. (to be added)

To deactivate this environment simply type:

``conda deactivate``

### Usage
------------
import keyfi as kf

import canteraKTH as ckth

from NN import *
