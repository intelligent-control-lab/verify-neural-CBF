# verify-neural-CBF
This is the official code for CoRL 2024 work "Verification of Neural Control Barrier Functions with Symbolic Derivative Bounds Propagation".

## *Update: Python implementation of training neural CBF*
**We have released the Python implementation of data collection, neural CBF model training, and visualization to further facilitate the research community. Checkout [Python Implementation](./python#python-implementation) for details.**

## Preparation
The code is based on [Julia](https://julialang.org/) and is tested with Julia v1.9.4. Check [here](https://julialang.org/downloads/oldreleases/) to install Julia environment. Install `ModelVerification.jl` from [this repo](https://github.com/intelligent-control-lab/ModelVerification.jl) and check out the branch `verify_gradient` [here](https://github.com/intelligent-control-lab/ModelVerification.jl/tree/verify_gradient). Install `RobotZoo.jl` from [this repo](https://github.com/RoboticExplorationLab/RobotZoo.jl) and `TaylorModels.jl` from [here](https://github.com/JuliaIntervals/TaylorModels.jl). 

## Data Collection
To collect data for each robot dynamics, see Jupyter file `collect_data.ipynb` for details.

## Model training 
For the model training under Point Robot, see Jupyter file `train_naive_point.ipynb` for regular training and `train_adv_point.ipynb` for adversarial training. For the model training under Dubins Car, see Jupyter file `train_naive_car.ipynb` for regular training and `train_adv_car.ipynb` for adversarial training. For the model training under Planar Quadrotor, see Jupyter file `train_naive_planar_quad.ipynb` for regular training and `train_adv_planar_quad.ipynb` for adversarial training.


## Verification of neural CBFs
For the verificaiton under Dubins Car, see Jupyter file `verify_car.ipynb`. Similarly, `verify_point.ipynb` is for point robot and `verify_planar_quad.ipynb` is for planar quadrotor. Replace the corresponding path with `naive` or `adv` for different pre-trained models. Also, $\alpha$ and number of grids per dimension can also specified for ablation study. For baselines, specify `max_iter=1` in `BFS` mehtod to specify NNCB-IBP and otherwise, it is for BBV baseline.  There is a tutorial example on `ModelVerification.jl` regarding verifying neural control barrier function [here](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/main/tutorial/tutorial.ipynb).

## Citation 
If you find the repo useful, please cite:

H. Hu, Y. Yang, T. Wei and C. Liu
"[Verification of Neural Control Barrier Functions with Symbolic Derivative Bounds Propagation](https://proceedings.mlr.press/v270/hu25a.html)", Conference on Robot Learning (CoRL). PMLR, 2024
```
@InProceedings{pmlr-v270-hu25a,
  title = 	 {Verification of Neural Control Barrier Functions with Symbolic Derivative Bounds Propagation},
  author =       {Hu, Hanjiang and Yang, Yujie and Wei, Tianhao and Liu, Changliu},
  booktitle = 	 {Proceedings of The 8th Conference on Robot Learning},
  pages = 	 {1797--1814},
  year = 	 {2024},
  volume = 	 {270},
  publisher =    {PMLR},
}
```


