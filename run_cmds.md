
- [No Push + Velocity Control](#no-push--velocity-control)
- [Jan 10 with Push + Position Control](#jan-10-with-push--position-control)
  - [trimesh\_push\_inter150](#trimesh_push_inter150)
  - [plane\_push\_inter150](#plane_push_inter150)
  - [heightfield\_push\_inter150](#heightfield_push_inter150)
- [HOW TO SETUP](#how-to-setup)

# ENVIRONMENT
> Ubuntu 20 + CUDA 12.1 + python 3.8
```zsh
conda create -n unitree-rl python=3.8 -y
conda activate unitree-rl

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

```

# Jan 11
```zsh
# 4070Ti
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="Jan11_commit_reward_MORE_DISCRETE_NO_STAIR_P_trimesh_512_nn_add_01_dummyReward" --run_name="obs_partial_height" --actor_height_map_accessibility="partially_masked"

# B14
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="Jan11_commit_reward_MORE_DISCRETE_NO_STAIR_P_trimesh_512_nn_add_01_dummyReward" --run_name="obs_full_height" --actor_height_map_accessibility="full"
```

# Jan 10

# With 5 dummy reward
## No Push + Position Control + trimesh Terrain + humanoid_gym_gait_and_energy
```zsh
# trimesh

############################################################################################################
# Training
## 4070Ti -- with G1RoughCfg.env.partially_masked_height_map = True
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="P_trimesh_no_push_large_nn_add_01_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --run_name="obs_partial_height"

## 4070Ti -- with G1RoughCfg.env.partially_masked_height_map = True | 00 dummy reward
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="track_x2_NO_STAIR_P_trimesh_no_push_large_nn_add_00_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --run_name="obs_partial_height"

## More discrete terrain + 005 dummy reward + with G1RoughCfg.env.partially_masked_height_map = True
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="track_x5_MORE_DISCRETE_NO_STAIR_P_trimesh_no_push_large_nn_add_05_reward_with_humanoid_gym_gait_energy_passHeight2Critic" --run_name="obs_partial_height"

## B14 -- with G1RoughCfg.env.partially_masked_height_map = False
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="P_trimesh_no_push_large_nn_add_01_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --run_name="obs_full_height"


# B14
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="track_x2_MORE_DISCRETE_NO_STAIR_P_trimesh_no_push_large_nn_add_005_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --run_name="obs_partial_height"

python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="track_x5_MORE_DISCRETE_NO_STAIR_P_trimesh_no_push_large_nn_add_005_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --run_name="obs_full_height"

############################################################################################################
# Play
## 4070Ti -- with G1RoughCfg.env.partially_masked_height_map = True
python legged_gym/scripts/play.py --task=g1 --experiment_name="P_trimesh_no_push_large_nn_add_01_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --load_run="Jan10_23-57-21_obs_partial_height"

## B14 -- with G1RoughCfg.env.partially_masked_height_map = False
python legged_gym/scripts/play.py --task=g1 --experiment_name="P_trimesh_no_push_large_nn_add_01_reward_with_humanoid_gym_gait_and_energy_passHeight2Critic" --load_run="Jan10_23-57-21_obs_full_height"
```