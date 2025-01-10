
- [Jan 10](#jan-10)
  - [trimesh\_push\_inter150](#trimesh_push_inter150)
  - [plane\_push\_inter150](#plane_push_inter150)
  - [heightfield\_push\_inter150](#heightfield_push_inter150)


# Jan 10

## trimesh_push_inter150
```zsh
# trimesh

# Training
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="trimesh_push_inter150"
# python legged_gym/scripts/train.py --task=g1 --experiment_name="trimesh_push_inter150"

# Store to "logs/trimesh_push_inter150/Jan10_02-26-21_"

# Play
python legged_gym/scripts/play.py --task=g1 --experiment_name="trimesh_push_inter150" --load_run="Jan10_02-26-21_"

```


## plane_push_inter150
```zsh
# trimesh

# Training
python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="plane_push_inter150"
# Store to "logs/plane_push_inter150/Jan09_xxx"

# Play
python legged_gym/scripts/play.py --task=g1 --experiment_name="plane_push_inter150" --load_run="Jan09_xxx"

```

## heightfield_push_inter150
```zsh
# trimesh

# Training
python legged_gym/scripts/train.py --task=g1 --experiment_name="heightfield_push_inter150"

python legged_gym/scripts/train.py --task=g1 --headless --experiment_name="heightfield_push_inter150"
# Store to "logs/heightfield_push_inter150/Jan09_xxx"

# Play
python legged_gym/scripts/play.py --task=g1 --experiment_name="heightfield_push_inter150" --load_run="Jan09_xxx"

```