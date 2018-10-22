# nips-2018-ai-for-prosthetics

### Results (as of competition deadline)

![Training converged erroneously after 1.6M timesteps of rollout/training](https://media.giphy.com/media/1gUqnaVepaILezzDps/giphy.gif)

- Time investment:  
    ~1 month of ~20 hours/week on nights & weekends.
- What I accomplished:  
    Unfortunately, this competition was heavy on software engineering, leaving me with barely any time remaining to work on the actual RL side of the project.  That being said, I'm sure if I could spend another month on it, the ratio would flip, and most of my time would be dedicated to RL.  I built a distributed training system with 3 pieces:
    - MongoDB: Used for storage of past timesteps as a replay buffer.
    - 1 training instance: a GPU instance in EC2 (I recommend smallest p2) running `Train Distributed.ipynb`, which pulls the latest model checkpoint from S3, trains the model by sampling timesteps from the MongoDB instance, and then updates the model checkpoint in S3.  It performs this task in an endless loop.  It also optionally performs Validation in the background by rolling out 10 episodes and logging the average cumulative reward.
    - N rollout instances: any number of compute instances (I recommend smallest c5) in EC2 running `Rollout Distributed.ipynb`, which pulls the latest model checkpoint from S3, performs episode rollouts with frame skipping and exploratory noise, and writes the recorded timesteps to the MongoDB instance.  They perform this task in an endless loop.
    
    Why this distributed system was necessary: 
    1. OpenSim, the simulator necessary for this competition, is not multiprocessing-friendly.  Even if you create one osim-rl environment per process, they seem to share a single OpenSim process, i.e. all of your threads report the same reward, which probably indicates interference.
    2. OpenSim is extremely slow compared to many other RL environments.  It runs much slower than real-time.  Because of this, if we want to train our model in a reasonable amount of time, we need to parallelize it.
    
    Why the p5 instance is recommended for training:
    - When we have an instance dedicated to training, we can start to take advantage of
    
    Why the c5 instance is recommended for rollout:
    1. OpenSim is slow enough to negate the cost-benefit of using a GPU for rolling out the model — you would only use the GPU <1% of the time, which means ~90% of your costs are wasted.  For this reason, I recommend rolling out on a CPU machine.  The reason I recommend the smallest c5 is because additional memory and CPU resources are not needed, since we can't (or I didn't know how to) parellize the OpenSim executable on a single machine.
    

### Setup

These instructions are intended to work on **Ubuntu 16.04**.  For MacOS, follow the setup instructions on the osim-rl repo, and I recommend using conda.

1. Download and extract OpenSim build from conda package.
    Choose download link from https://anaconda.org/kidzik/opensim/files (latest version for your OS)
    
    ```
    $ cd ~
    $ wget https://anaconda.org/kidzik/opensim/4.0.0/download/linux-64/opensim-4.0.0-py36_17.tar.bz2
    $ mkdir opensim-4.0.0
    $ cd opensim-4.0.0
    $ tar -xf ~/opensim-4.0.0-py36_17.tar.bz2
    $ export PATH=$PATH:/home/ubuntu/opensim-4.0.0/bin
    $ rm ~/opensim-4.0.0-py36_17.tar.bz2
    ```

1. Install python 3.6

    ```
    $ sudo add-apt-repository ppa:jonathonf/python-3.6
    $ sudo apt update
    $ sudo apt install python3.6 python3.6-dev
    ```

1. Install pip3.6

    ```
    $ cd ~
    $ wget https://bootstrap.pypa.io/get-pip.py
    $ sudo python3.6 get-pip.py
    $ rm get-pip.py
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
    
    Visit the link shown, and open `scripts/Verify Setup.ipynb`


### Setup for distributed training

1. Launch a MongoDB instance
    The easiest (but expensive) solution is to sign up for a hosted DB on mongodb.com.  I would expect to need the tier which costs $20-50/month depending on how many timesteps you collect.
    
    Another solution is to host the MongoDB on your GPU training instance, which has the advantage of eliminating network latency for your DB queries on this machine, enabling you to take better advantage of your GPU.  Note that you will need to allow inbound network traffic on port `27017`, as well as pass the `--bind_ip_all` option when starting MongoDB. I couldn't get both "auth" and "bind_ip_all" options working using `/etc/mongod.conf`, so I ended up just manually running `sudo mongod --auth --bind_ip_all &` in a `screen`.
    
    1. Create a user/password (might be created for you if you're hosting on mongodb.com)
    1. Create a `ProstheticsEnv` database with a `timesteps` collection and an `events` collection.
1. Launch 1 or more c5 instances on EC2 on Ubuntu 16.04.
    1. Follow the repo setup/install instructions (above).
    1. Create a `.env` file with your MongoDB hostname and credentials.
        ```
        $ cp .env.example .env
        $ vim .env
        ```
    1. First, run the `Rollout Random.ipynb` notebook to seed the replay buffer (MongoDB) with some random exploration of the state+action space.
1. Create an S3 bucket for your model checkpoints and configure it to be public read/write.
1. Launch a p2 instance on EC2 using the Deep Learning Base AMI on Ubuntu.
    1. Follow the repo setup/install instructions (above).
    1. Point the cuda symlink (in `/usr/local/cuda`) to the desired cuda version if it's not already.
        ```
        $ ls /usr/local/cuda
        $ sudo rm /usr/local/cuda
        $ sudo ln -s /usr/local/cuda-9.2 /usr/local/cuda
        ```
    1. Create a `.env` file with your MongoDB hostname and credentials.
        ```
        $ cp .env.example .env
        $ vim .env
        ```
    1. Open the `Train Distributed.ipynb` notebook.
    1. Update the location of your S3 bucket.
    1. Comment out the part where it loads the initial model checkpoint from S3 (since you won't have a checkpoint yet).
    1. Run the notebook indefinitely.
    1. Uncomment the part where it loads the model checkpoint from S3, so that if you restart this notebook in the future, you will continue training where you left off.
1. Go back to your (1 or more) c5 instances.
    1. Update the location of your S3 bucket.
    1. Run the `Rollout Distributed.ipynb` notebook indefinitely.



