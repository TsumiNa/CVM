# CVM

CVM is a python package for solution limit calculation using **Cluster Variation Method** (CVM).

## Concept

![concept](https://github.com/tsumina/cvm/raw/master/docs/_static/conceput.png)

## Installation

1. Download `CVM` source codes.

   ```bash
   $ git clone https://github.com/TsumiNa/CVM.git cvm
   ```

2. Go into the downloaded dir.

   ```bash
   $ cd cvm
   ```

3. Create virtual environment for `CVM` and activate it. We are assuming you have installed the `miniconda`.

   ```bash
   $ conda env create -f conda_env.yml
   $ conda activate cvm
   ```

4. Install `CVM` locally.

   ```bash
   $ pip install -e .
   ```

5. See also our sample.
   ```bash
   $ cd samples
   $ jupyter lab
   ```
