# LUMIA

Full code will be realeased uppon aceptance.
This guide explains how to set up the environment and install the necessary dependencies for this project.

## Prerequisites

Ensure you have the following installed:
- **Anaconda** or **Miniconda**
- **Python 3.10.14**

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <https://anonymous.4open.science/r/LUMIA-6DA3/>
cd <lumia>
```

### 2. Create new environment
Create a new environment:
```bash
conda create -n lumia python=3.10.14 -y
conda activate lumia
```

### 3. Install requirements
Install requirements:
```bash
pip install -r requirements.txt
```

### Using automated script.
Change exec permission and execute automated script:

```bash
sudo chmod +x setup_environment.sh
./setup_environment.sh
```

### Running.
To run part of the experiments execute the following scripts:

For unimodal experiments:

```bash
sudo chmod +x setup_environment.sh
./run_unimodal_experiment.sh
```
For multimodal experiments:
```bash
sudo chmod +x setup_environment.sh
./run_multimodal_experiment.sh
```
