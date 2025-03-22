[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7038247.svg)](https://doi.org/10.5281/zenodo.7038247)
# SKA Science Data Challenge-1, tutorial/hack format

## Description

Welcome to the SKA Science Data Challenge-1! The SKA planned for 4 data challenges to prepare the scientific community for the challenges that the SKA will present. We took the 1st data challenge and turned it into a tutorial format suitable for early carer researchers. From the tutorials, you will learn the following:

- *Tutorial 1* : Data preprocessing
- *Tutorial 2* : Source classification

The aim of the tutorials is as  follows:
  - Source finding (RA, Dec) to locate the centroids and/or core positions,
  - Source property characterization (integrated flux density, possible core fraction, major and minor axis size, major axis position angle)
  - Source classification (one of SFG, AGN-steep, AGN-flat)


## Data
3 different simulated data are used in this workflow, where the simulation represents the following frequencies:
- 560 MHz
- 1400 MHz
- 9200 MHz

A  sample of each image can be downloaded using the script `binder/download_sample_data.sh`, to excute the script run the following:

```bash
>  bash binder/download_sample_data.sh
```


## Hackathon Task
From the proposed pipeline, investigate new ways to find/classify sources.


## Prerequisites

All the libraries/dependencies necessary to run the tutorials are listed in the [requirements.txt](https://github.com/ngoosojk/dataChallengeRepo/blob/master/requirements.txt) file.


## Installation

All the required libraries can be installed using pip and the [requirements.txt](https://github.com/ngoosojk/dataChallengeRepo/blob/master/requirements.txt) file in the repo:

```bash
> pip install -r requirements.txt
```

### Would you like to clone this repository? Feel free!

```bash
> git clone https://github.com/ngoosojk/dataChallengeRepo.git
```

Then make sure you have the right Python libraries for the tutorials. 


### New to Github?

The easiest way to get all of the lecture and tutorial material is to clone this repository. To do this you need git installed on your laptop. If you're working on Linux you can install git using apt-get (you might need to use sudo):

```
apt install git
```

You can then clone the repository by typing:

```
git clone https://github.com/ngoosojk/dataChallengeRepo.git
```

To update your clone if changes are made, use:

```
cd dataChallenge_hack/
git pull
```

-----

