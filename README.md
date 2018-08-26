# nips-2018-ai-for-prosthetics


### Setup

1. Download and extract OpenSim build from conda package.
    Choose download link from https://anaconda.org/kidzik/opensim/files (latest version for your OS)
    
    ```
    $ cd ~
    $ wget https://anaconda.org/kidzik/opensim/4.0.0/download/linux-64/opensim-4.0.0-py36_17.tar.bz2
    $ mkdir opensim-4.0.0
    $ cd opensim-4.0.0
    $ tar -xf ~/opensim-4.0.0-py36_17.tar.bz2
    ```

1. Install python 3.6

    ```
    $ sudo add-apt-repository ppa:jonathonf/python-3.6
    $ sudo apt update
    $ sudo apt install python3.6 python3.6-dev
    ```

1. Clone the repo

    ```
    $ cd ~
    $ git clone https://github.com/colllin/nips-2018-ai-for-prosthetics.git
    ```
    
1. Install pipenv

    ```
    $ pip3.6 install --user pipenv
    ```

1. Create a virtual environment & install dependencies

    ```
    $ cd nips-2018-ai-for-prosthetics
    $ pipenv install
    ```

1. Create a Jupyter kernel to run code in this virtual environment

    ```
    $ pipenv run python -m ipykernel install --user --name prosthetics
    ```
    
1. Launch a Jupyter notebook server and open a notebook

    ```
    $ jupyter notebook &
    ```
    
    Visit the link shown, and open `Verify Setup.ipynb`


### Development





