
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

# Jan 10

# With 5 dummy reward
## No Push + Position Control + trimesh Terrain + humanoid_gym_gait_and_energy
```zsh
# trimesh

# Training
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="P_trimesh_no_push_large_nn_add_5_reward_with_humanoid_gym_gait_and_energy"

# Play
python legged_gym/scripts/play.py --task=g1 --experiment_name="P_trimesh_no_push_large_nn_add_5_reward_with_humanoid_gym_gait_and_energy" --load_run="Jan10_17-06-33_"
```