# h2o2.ai presents: Voice of H2O
## A production-ready web application for Voice of Customer (VOC) analysis

From retail to big tech, customers can be hard to read - and that's where we come in. Introducing "Voice of H2O", our cutting-edge web application for Voice of Customer analysis. Built on H2OWave, our intuitive interface and advanced analytical capabilities allow firms to bridge the gap between big data and individual customers.

Our application allows you to:
1. Explore key themes in customer feedback
2. Visualise sentiments of customers
3. Perform trend analysis over time

Whether you're a small business owner or a large enterprise, our VoC analysis application will help you improve customer satisfaction, boost loyalty, and drive business growth. And better yet - you can do this all for free. Simply follow setup instructions to containerise and build our web application on your very own device.

## Setup Instructions

### Option 1: Docker

#### Deployment

To build Docker image and container, run the following commands on h2o2.ai project folder:

```bash
docker compose up
```

#### Running Pipeline via Notebook

Note that you will need to have the necessary computing power to run model training on Docker.

To run all pipelines via notebook, run the following commands on final_presentation folder:

```bash
python3 run_notebook.py
```

#### Running Pipeline via Terminal

Note that you will need to have the necessary computing power to run model training on Docker.

To run all pipelines via terminal, run the following commands on h2o2.ai project folder:

```bash
# Preprocess training data. This step should be done before calling train functions.
python3 -m src.preprocessing.transformations

# Train sentiment analysis models
python3 -m src.models.sentiment_analysis.train.train

# Train topic modelling models
python3 -m src.models.topic_modelling.train.train

# Predict test data. Test file path should be supplied from config file.
python3 -m src.models.predict

# Execute unit testing.
python3 -m src.unittest.unit_testing
```

#### Running App

Open the app on http://127.0.0.1:8080/ or http://localhost:8080/

### Option 2: EC2 Instance

Unfortunately, it's not possible to run H2O Wave app on EC2 Instance given the limited access to AWS provided on RLCatalyst as it needs further setup on AWS to open the relative port of EC2 Instance (https://h2o.ai/blog/deploy-a-wave-app-on-an-aws-ec2-instance/).

Hence, EC2 Instance is only used for model training and test prediction purposes.

#### Prerequisites

You will need to have an ubuntu EC2 GPU instance running and connect to the instance via terminal.

Then, copy all files to the instance.

#### Environment Setup

To setup the EC2 instance, run the following commands on h2o2.ai project folder:

```bash
# Install pip for Python 3
sudo apt install python3-pip

# Install all libraries needed
pip install -r requirements.txt
```

#### Running Pipeline via Notebook

To run all pipelines via notebook, run the following commands on final_presentation folder:

```bash
python3 run_notebook.py
```

#### Running Pipeline via Terminal

To run all pipelines via terminal, run the following commands on h2o2.ai project folder:

```bash
# Preprocess training data. This step should be done before calling train functions.
python3 -m src.preprocessing.transformations

# Train sentiment analysis models
python3 -m src.models.sentiment_analysis.train.train

# Train topic modelling models
python3 -m src.models.topic_modelling.train.train

# Predict test data. Test file path should be supplied from config file.
python3 -m src.models.predict

# Execute unit testing.
python3 -m src.unittest.unit_testing
```

### Option 3: Local

#### Prerequisites

You will need to have a valid Python and Conda installation on your system.

#### Environment Setup

To avoid library dependency issues, run the following commands on h2o2.ai project folder:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate created conda environment
conda activate voc_env
```

#### Running Pipeline via Notebook

Note that you will need to have the necessary computing power to run model training on local.

To run all pipelines via notebook, run the following commands on final_presentation folder:

```bash
python run_notebook.py
```

#### Running Pipeline via Terminal

Note that you will need to have the necessary computing power to run model training on local.

To run all pipelines via terminal, run the following commands on h2o2.ai project folder:

```bash
# Preprocess training data. This step should be done before calling train functions.
python -m src.preprocessing.transformations

# Train sentiment analysis models
python -m src.models.sentiment_analysis.train.train

# Train topic modelling models
python -m src.models.topic_modelling.train.train

# Predict test data. Test file path should be supplied from config file.
python -m src.models.predict

# Execute unit testing.
python -m src.unittest.unit_testing
```

#### Running App

To start app, run the following commands on h2o2.ai project folder:

```bash
wave run src.app.app
```

Open the app on http://localhost:10101/
