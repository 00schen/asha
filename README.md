## Setup Instructions ##
1. Create a separate conda environment, and install the required packages using `pip install -r requirements.txt`. You 
   may need to install additional dependencies to install the `dlib` package correctly, which is necessary for using
   real webcam gaze inputs.
2. To use real gaze inputs, download the file from `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`
   and extract it to `image/gaze_capture/model_files/`.
3. Run the following commands to install local packages:
    - `pip install -e rlkit/`
    - `pip install -e image/`
    - `pip install -e assistive-gym/`
4. Pretrained models from phase 1 (as described in the paper) are located in `image/util_models`. 
5. To train your own versions of these models:
   - Run`python image/rl/scripts/sac_pretrain_(oneswitch/bottle).py --no_render` to pretrain a model, choosing between
     `oneswitch` or `bottle` for the desired environment. 
   - To use GPUs for training, add the flags `--use_ray --gpus N_GPUS`, where `N_GPUS` is the number of GPUs used for 
     training.
   - The model will be saved to `image/logs/pretrain-sac-(oneswitch/bottle)/EXPERIMENT_NAME/`, where `EXPERIMENT_NAME`
     will be a folder name specific to the experiment ran, containing the date and time of when the experiment was
     started.
   - The final model parameters will be located in the folder as `params.pkl`, and this is the file corresponding to the
     included `(OneSwitch/Bottle)_params_s1_sac.pkl` file in `image/util_models`.
   - To pretrain a model using a deterministic encoder (for the described ablation experiments in the paper), add the
     flag `--det`. This will produce a model file corresponding to the included
     `(OneSwitch/Bottle)_params_s1_sac_det_enc.pkl` file in `image/util_models`.
   - To use your own pretrained files, you should replace the corresponding included files with them.
   - To pretrain a model for the Bottle environment, you should first collect some demonstrations to bootstrap RL using
     `python image/rl/scripts/collect_demo.py --no_render`. The demonstrations will be saved to `image/demos`,
     and will be loaded from here automatically by the pretraining script for Bottle. 
6. To run experiments with real webcam gaze input, using the full ASHA method in the default experimental setup
   described in the paper, run `python image/rl/scripts/sac_experiment_s2-calibrate.py --env_name (OneSwitch/Bottle)
   --exp_name SAVE_DIR`.
   - To change the number of episodes (default 50), add the flag `--epochs N_EPISODES`, where `N_EPISODES` is the
     desired number of episodes.
   - For both environments, add the flag `--mode no_online` to use the non-adaptive baseline.
   - In the Switch environment, to avoid calibrating on the second to the right switch, add the flag `--mode no_right`.
   - In the Bottle environment, to avoid calibrating on episodes where the sliding door needs to be opened first,
     add the flag `--mode no_door`.
7. To run simulated ablation experiments, run:
   - `python image/rl/scripts/sac_experiment_s2-calibrate.py --env_name (OneSwitch/Bottle) --epochs 100 --sim
     --no_render --exp_name SAVE_DIR` to use the full ASHA method.
   - To use the random latent baseline, add the flag `--rand_latent`.
   - To use the non-adaptive baseline, add the flag `--mode no_online`.
   - To use a deterministic online input encoder, add the flag `--det`.
   - To use the model pretrained using a deterministic encoder, add the flag `--pre_det`.
   - To not train on relabeled failure episodes, add the flag `--no_failures`.
   - To train using latent regression, add the flag `--latent_reg`.
   - To use SAC from scratch, instead run `python image/rl/scripts/sac_experiment_s2-vanilla.py
     --env_name (OneSwitch/Bottle) --epochs 100 --sim --no_render --exp_name SAVE_DIR`
   - To use GPUs and automatically use 10 different seeds, add the flags `--use_ray --gpus N_GPUS`, where `N_GPUS` is
     the number of GPUs used for training. Otherwise, to use different seeds, you will need to run the scripts once
     for each seed, and change the `seedid` list param in the `search_space` dict of each script so that the first 
     element of the list is the desired seed.
   
8. For both real gaze and simulated experiments, - The experiment results will be saved to
   `image/logs/SAVE_DIR/EXPERIMENT_NAME`, where `EXPERIMENT_NAME` will be a folder name specific to the experiment ran,
   containing the date and time of when the experiment was started. When multiple seeds are ran at once, they are saved
   to different folders. In the save folder, metrics are logged to `metrics.pkl`, and the episodes are logged to 
   `data.pkl`. 

## Acknowledgements
Contains code, models, and data from the PyTorch implementation of "Eye Tracking for Everyone,"
found at https://github.com/CSAILVision/GazeCapture/tree/master/pytorch.
> Kyle Krafka, Aditya Khosla, Petr Kellnhofer, Harini Kannan, Suchi Bhandarkar, Wojciech Matusik and Antonio Torralba.
> “Eye Tracking for Everyone”. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Contains code and models from "Assistive Gym: A Physics Simulation Framework for Assistive Robotics," found at
https://github.com/Healthcare-Robotics/assistive-gym. The original environments and corresponding models are not used, but underlying environment set up is borrowed.
> Z. Erickson, V. Gangaram, A. Kapusta, C. K. Liu, and C. C. Kemp, “Assistive Gym: A Physics Simulation Framework for
> Assistive Robotics”, IEEE International Conference on Robotics and Automation (ICRA), 2020.

Contains code from RLkit, found at https://github.com/vitchyr/rlkit. 