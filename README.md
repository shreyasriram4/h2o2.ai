# h2o2.ai presents: Voice of H2O

## A production-ready web application for Voice of Customer (VOC) analysis

From retail to big tech, customers can be hard to read - and that's where we come in. Introducing "Voice of H2O", our cutting-edge web application for Voice of Customer analysis. Built on H2OWave, our intuitive interface and advanced analytical capabilities allow firms to bridge the gap between big data and individual customers.

Our application allows you to:
1. Explore key themes in customer feedback
2. Visualise sentiments of customers
3. Perform trend analysis over time

Whether you're a small business owner or a large enterprise, our VoC analysis application will help you improve customer satisfaction, boost loyalty, and drive business growth. And better yet - you can do this all for free. Simply follow setup instructions to containerise and build our web application on your very own device.

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

cliftonfelix
hadragon26
leeweiqing
madelinelimm
shreyasriram4

## Acknowledgments

 - Hat tip to anyone who's code was used
 - Inspiration
 - etc.
