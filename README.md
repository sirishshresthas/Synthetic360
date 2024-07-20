# Synthetic Data Generation for Benchmark for Managers

![Synthetic Data](images/sdv.png)

## Table of Contents
- Introduction
- Getting Started
  - Prerequisites
  - Installation
- Usage
  - Configuration
  - Generating Synthetic Data
- Roadmap
- Contributing
- License

## Introduction
In leadership development, the Benchmark for Managers 360-degree feedback assessments serves as a crucial resource, encompassing a wide range of sensitive and confidential information pivotal for understanding leader's development path. However, the sensitive nature of this data imposes significant restrictions on its direct dissemination, utilization for various analytical and developmental purposes, and data collection. To navigate these privacy concerns effectively, our solution employs synthetic data generation techniques. These methodologies enable us to craft artificial datasets that replicate the statistical properties of the original data without compromising individual privacy. This innovative approach not only ensures compliance with data protection standards but also extends the dataset's applicability across a multitude of projects, including reporting and machine learning modeling. This repository is dedicated to offering a bespoke synthetic data generation solution tailored specifically for the Benchmark for Managers dataset, leveraging the power of Generative Adversarial Network, a deep learning based AI model.

## Getting Started

### Prerequisites
To embark on generating synthetic data, ensure you have the following prerequisites in place:
- Python version 3.11 or higher.
- A virtual environment setup is highly recommended for an isolated development environment. We suggest using Conda for this purpose, which can be installed following the instructions at [Conda's official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

### Installation
Begin by cloning this repository to your local environment. Once cloned, all necessary dependencies can be installed either through Conda or pip, depending on your preference. These dependencies are meticulously listed in either `conda.yml` for Conda users or `requirements.txt` for pip users.

For Conda:
```bash
conda env create -n vsynth -f conda.yml
conda activate vsynth
```

For pip users:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration
Before proceeding with the generation of synthetic data, it's crucial to configure your environment correctly. This process involves setting up several environment variables that are essential for integrating with the Azure ecosystem, which our solution heavily relies on. These variables include:

- `ENV`: Specifies the environment (e.g., "dev" for development).
- `AZURE_STORAGE_ACCOUNT_NAME`: The name of your Azure Storage account.
- `AZURE_STORAGE_ACCOUNT_KEY`: Your Azure Storage account key.
- `AZURE_BLOB_ENDPOINT`: The endpoint for your Azure Blob storage.

These variables can be set up in a `.env` file at the root of the project for easy management.

### Generating Synthetic Data
The primary notebook to initiate the synthetic data generation process is `app.ipynb`. This notebook serves as the cornerstone of our process, preparing and sharing the initial dataset across other components of the project.

To generate synthetic data leveraging the Gaussian Copula method, run the `copula_synthesizer.ipynb` notebook. For those interested in utilizing the CTGAN method, the `ctgan_synthesizer.ipynb` notebook is available. Both notebooks are designed to seamlessly integrate with the data prepared in `app.ipynb`, ensuring a cohesive and efficient workflow.

## Roadmap
The roadmap section will outline planned enhancements and feature additions to the synthetic data generation project. This may include improvements in data synthesis methodologies, expansion of dataset coverage, and integration with additional tools and platforms other than the Azure ecosystem.


## License
This project is released under a specific license, which can be found in the LICENSE file. We encourage you to review the license terms before using or contributing to the project.

By adhering to these guidelines and utilizing the provided resources effectively, users can generate high-quality synthetic data for the Benchmark for Managers dataset, fostering innovation while ensuring compliance with privacy standards.