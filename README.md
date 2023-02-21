# h2o2.ai: DSA4263 (Sense-making Case Analysis: Business and Commerce) project

Building a production-ready web application for Voice of the Customer (VOC) analysis.

## Folder Structure

```bash
.
├──app/
│   └── pages/
├──data/
│   ├── raw/
│   └── processed/
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── figures/
├── README.md
├── reports/
├──requirements.txt
├── results/
├── src/
│   ├──__init__.py <- Make src a Python module
│   ├──preprocessing/
│       └── preprocressing_utils.py <- Scripts to preprocess corpus
│   ├──models/
        ├──sentiment_analysis/
        └──topic_modelling/
│   └──visualisations/
│       └── eda_utils.py <- Scripts to run visualisations
├── .dockerignore
└── .gitignore 

```

## Prerequisites

You will need to have a valid Python and Conda installation on your system.

## Git Flow
 - Branch off main and do dev work, remember to git pull origin main
 - Create PR to merge to main once done and delete that branch
 - For bugfixes, if the branch is already merged, create hotfix branch based off main
 - Create PR to merge to prd when ready to realease and changes in release

## Setup Instructions

### Option 1: Without Docker
For installation, first close this repository, and generate the virtual environment required for running the programs within your IDE of choice:

```bash
#Create environment from file
conda env create -f environment.yml

#Activate created conda environment
conda activate voc_env

```

### Option 2: With Docker

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

 - Python 3.9

## Authors

madelinelim - Initial work (2021)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - Hat tip to anyone who's code was used
 - Inspiration
 - etc.

# IN PROGRESS
