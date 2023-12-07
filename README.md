# WineELO

In this repository, we explore an alternative to star ratings systems: the WineELO score. We scrape user (star) ratings from wine review platform Vivino, and convert them to WineELO scores to explore the merit of this new metric. 

This repository contains the following files:

- **wine_elo.ipynb**: notebook containing analysis
- **wine_data_cleanup.py**: python module with custom functions used in notebook
- **web_scraper.py**: web scraper used to retrieve Vivino user reviews

The dataset of scraped Vivino ratings have been been omitted from this repository due to size constraints. You may scrape data yourself by running web_scraper.py. Please note that you will need to download the appropriate version of chromedriver (https://chromedriver.chromium.org/downloads) to run this script. 

## Technologies
- Python
- Jupyter Notebook

## Getting Started

### Setting Up Your Python Environment with Conda

To ensure a consistent and isolated development environment for our project, we use Conda, a popular package and environment manager. Here’s how you can set up and manage your Conda environment:

#### 1. Creating the Environment
First, we create a Conda environment based on our project's dependencies, which are listed in a YAML file (`conda_env.yml`). This file specifies the exact versions of Python and other libraries we need.

**Command:**
```bash
conda env create -f conda_env.yml
```
*What it does*: This command reads the `conda_env.yml` file and creates a new environment named `sommsense` (the name is usually defined within the YAML file) with all the specified packages installed.

#### 2. Updating the Environment
If there are updates to our project dependencies, reflected in changes to the `conda_env.yml` file, you'll need to update your environment to match these new specifications.

**Command:**
```bash
conda env update -f conda_env.yml
```
*What it does*: This command updates the `sommsense` environment, adding, removing, or updating packages as per the revised `conda_env.yml` file.

#### 3. Activating the Environment
Before starting work on the project, you need to activate the environment. This step is crucial to ensure that you are using the correct Python interpreter and libraries specific to our project.

**Command:**
```bash
conda activate sommsense
```
*What it does*: Activates the `sommsense` environment. Once activated, any Python command you run will use the environment’s settings and packages.

---

**Note**: Always ensure that you're in the correct Conda environment before running or developing the project. This helps in maintaining consistency across different development setups and reduces compatibility issues.



1. Clone this repo
2. Install the appropriate version of chromedriver and the executable to your PATH
3. Run web_scraper.py to get a full and fresh set of Vivino wine reviews
4. Run wine_elo.ipynb to run the analysis notebook

## Authors

Roald Schuring

