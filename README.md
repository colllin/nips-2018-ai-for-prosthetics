# nips-2018-ai-for-prosthetics

This repo contains my work for the NIPS 2018 [AI for Prosthetics Challenge](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge).

## Post-mortem

![Training converged erroneously after 1.6M timesteps of rollout/training](https://media.giphy.com/media/1gUqnaVepaILezzDps/giphy.gif)

### Time investment
~1 month of ~20 hours/week on nights & weekends.

### What I accomplished
This competition was heavy on software engineering, which unfortunately left me with barely any time remaining to work on the actual RL side of the project.  That being said, I'm sure if I could spend another month on it, the ratio would flip, and most of my time would be dedicated to RL.
- I hooked up a model to the simulator environment and implemented observation hacking, reward hacking, action "binning", frame skipping, and exploratory noise. 🎉
- I built a distributed training system to reduce the time and cost of training. 🙌
- I ran two training experiments using a recent variant of DDPG. 💩

### 2 Experiments
Both experiments used the same custom observation (see `environment/observations.py`):
- Every body position value from the current observation state dict, transformed to be relative to the center of mass's absolute position.
- Every body velocity value from the current observation state dict, transformed to be relative to the center of mass's absolute velocity.
- Every body acceleration value from the current observation state dict, transformed to be relative to the center of mass's absolute acceleration.
- Every other value from the environment's current observation dict, including joint orientations, rotations, and accelerations, muscle forces and fiber lengths, etc.
- The center of mass's velocity relative to the target velocity (e.g. `[-1, 0, 0]` if the model is 1 m/s slower than the target velocity along the x axis).  This was intended to be a stepping stone to the second phase of the competition, where the target velocity is variable, and we need to communicate the instantaneous target velocity to the model. 
- The highest-order observation values from the past three timesteps. For example, I would exclude body position and velocity vectors from the previous timestep since those values are accounted for in the current observation's velocity and acceleration vectors. Then the previous timestep's acceleration vector combined with the current timestep's pos,vel,acc vectors mean the pos,vel information from 2 timesteps ago is not interesting information.  The idea in including the previous timesteps is to allow the model to compute higher-order derivatives.  The idea of dropping most of the previous information was mostly just to reduce the observation size, which still ended up at 1,260 values.

Both experiments (intended to have) used the same action handling (see `environment/actions.py`):
- Binning into 2 bins (binary actions, either 0 or 1). This idea came from the winning solution from last year's **Learn to Run** challenge.
- Frame skipping during rollout
- Exploratory noise during rollout (see `Rollout Distributed.ipynb`)

#### Experiment 1: Custom rewards + episode "done" hacking:
- I had a huge custom rewards function (see `environment/rewards.py`) which I probably never tested enough, but it was intended to essentially "coach" the policy into expected running mechanics.  My understanding was that the competition was meant to aid prosthetics research by helping designers understand how their devices would affect and enable the mechanics of walking and running.  Therefore, we are not really interested in the RL model inventing totally new running mechanics.
- I had a function which was intended to detect scenarios which are unlikely to result in a positive outcome, and abort the episode early. For example, if the head went backwards more than 10cm, or more than 50cm outside the intended direction of travel, I aborted the episode early.
- I ran the distributed training for about 800k timesteps, and then stopped it because it appeared that the model had converged on a poor solution — each rollout episode was producing an identical result.
- I analyzed the timesteps in the database and found some strange numbers right away — most obviously, the observation showed `y` positions (vertical axis) on a large number of timesteps to be *negative*.  Was there a bug in the simulator? Had I configured the simulator's `integrator_accuracy` to be too low (too high of a convergence threshold)?  Looking at more of the data, I found that a large number of the action values were `-1`, but the valid range of action values is `[0,1]`. Clearly there is a bug in my code for "binning" the actions, which was intended to be binary for this experiment (only ever 0 or 1).

#### Experiment 2: Default rewards + no episode hacking + fixed `-1 action` bug
- I threw out all the timesteps from before.
- I threw out the custom rewards function since I didn't have time to debug it, and it was fairly complicated, and could have been causing more problems than benefits.
- I fixed the `-1 action` bug.
- I removed the episode hacking, because I was concerned that it might negatively impact training due to the model's estimation of future rewards. Essentially, by aborting the episode early without also coordinating a large negative reward, the model might capitalize on a sweet spot which maximizes the short term reward before aborting the episode.  Comparatively, if the model performs the goal behavior but I then allow it to fall forward for a dozen timesteps, resulting in a large negative cumulative reward, it might be deterred from exploring this behavior further.
- I ran ~25k steps of random exploration to seed the database.
- I ran the distributed training for 1.6M timesteps, at which point I killed the training, because, again, the model appeared to converge at an undesirable behavior (as seen in the GIF above).

### That distributed training system, though.
The distributed training system has 4 components:
- MongoDB: Used for storage and sharing of past timesteps as a replay buffer.
- S3: Used for storage and sharing of model checkpoints.
- 1 training instance: a GPU instance in EC2 (I recommend smallest p2) running `Train Distributed.ipynb`, which pulls the latest model checkpoint from S3, trains the model by sampling timesteps from the MongoDB instance, and then updates the model checkpoint in S3.  It performs this task in an endless loop.  It also optionally performs Validation in the background by rolling out 10 episodes and logging the average cumulative reward.
- N rollout instances: any number of compute instances (I recommend smallest c5) in EC2 running `Rollout Distributed.ipynb`, which pulls the latest model checkpoint from S3, performs episode rollouts with frame skipping and exploratory noise, and writes the recorded timesteps to the MongoDB instance.  They perform this task in an endless loop.
    
#### Why this distributed system was necessary: 
1. OpenSim, the simulator used for this RL environment, is extremely slow compared to many other RL environments.  It runs much slower than real-time.  Because of this, if we want to train our model in a reasonable amount of time, we need to parallelize it.
1. OpenSim is not multiprocessing-friendly.  Even if you create one osim-rl environment per process, they seem to share a single OpenSim process, i.e. all of your threads report the same reward, which probably indicates interference.
Thus, we need to parallelize across multiple machines.
    
#### Why the p2 instance is recommended for training:
When we have an instance dedicated to training, we can take advantage of the GPU since we no longer need to wait for the slow RL environment.
    
#### Why the c5 instance is recommended for rollout:
OpenSim is slow enough to negate the cost-benefits of using a GPU for rolling out the model — you would only use the GPU <1% of the time, which means ~90% of your costs are wasted.  For this reason, I recommend rolling out on a CPU machine.  The reason I recommend the smallest c5 is because the single environment instance only uses a fraction of the available CPU and memory resources, and we can't (or I didn't know how to) parellize the OpenSim executable on a single machine.
    
#### Why MongoDB?
- It's a simple NoSQL storage solution which makes the DevOps burden very light. 
- The querying features would be useful even if we were running this entire operation on a single machine and didn't need the asynchronous communication mechanism.
- We would need a scalable data storage solution anyway, since I ran into memory issues reading/writing CSVs after only about 20k timesteps, and we likely need millions to solve this problem.
- Bonus features like MongoDB Compass GUI for exploring and analyzing the data.
- Centralized sharing of timesteps and logs across machines.
    
#### Why S3?
Centralized location to read & write the latest model checkpoint for coordination across machines.

#### Cost of training (theoretical)
- MongoDB: $0. I ended up hosting the MongoDB instance on the GPU training instance, which has the advantage of reducing costs as well as eliminating network latency when sampling timesteps from the database.
- S3: This is probably negligible — the GPU training node is reading 1 checkpoint and writing 2 checkpoints every ~7 minutes, and the rollout nodes are each reading 1 checkpoint every ~3 minutes.
- GPU node: the smallest EC2 p2 instance at time of writing costs roughly $1/hour on-demand. This instance trained on 2-3 million timesteps per hour. This is important because you want to pace it with the rate of rollout and collection of new timesteps.
- CPU node: the smallest EC2 c5 instance at time of writing costs roughly $.10/hour on-demand, but I ran them as a fleet of spot instances to reduce costs by as much as 50%.  Each c5 instance rolled out about 20,000 timesteps per hour, or about 500,000 timesteps per 24 hours.

On the second experiment, I trained for 24 hours with one on-demand p2 instance and a fleet of 3 c5 spot instances, capturing about 1.6 million timesteps in that time.  The number of spot instances should be adjusted based on where you are in your training.  I didn't want to capture too many early episodes before the model was starting to explore a more interesting solution space.  The p2 instance cost roughly $24 for the 24 hours, and the c5 spot instances cost roughly $.05 × 24 × 3 = $4.  This is about $28/day, or far less than I expected from reading post-mortems from the **Learn to Run** challenge from 2017.  Since my training didn't result in a good solution, I can't say how many timesteps would be needed to fully train the model, or what the total cost would be.

### Next steps
- Explore the hyperparameters of the model, especially the learning rate (using a "learning rate finder") and the layer dimensions within the policy architecture. Again, I didn't even really get to these obvious parts of the process, since I needed to spend so much time just to get it training at a reasonable cost and speed.
- Implement a prioritized replay buffer to better focus the training on the most interesting known parts of the solution space.  This would allow me to make better use of my GPU instance by running more rollout nodes simultaneously, since I know that the more recent or better episodes will be prioritized, and don't need to worry that more random episodes will dilute the signal from more interesting observation/action spaces.
- Finish validating and implementing the reward hacking to "coach" the model into reasonable running form.  I would probably take a more iterative approach next time — train the model some, see where it's going wrong, e.g. the legs are crossing each other!  Then implement a reward (or penalty) to teach it to avoid this behavior.

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



