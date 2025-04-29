# 25/04/20256

AC 
- implement TD(lambda) with forward-view : ok
- save controlled trajectories to train LatentODE : ok

LatentODE 
- make it work with collected controlled trajectories : it works but very badly

Theory
- in-depth study of Yildiz code : ok

LateX
- design the end-to-end algorithm : ok
- solve the interaction with real world issue : ok

# 28/04/20256

LatentODE 
- clean the code : ok
- add identity and laplace encoders : ok

# 28/04/20256

LatentODE 
- identify why performance is terrible : ok
- clean and modularize the LatentODE code : ok
- add interpolation for continuous control : ok
- add visualization : ok

# TODO 

AC 
- modularize Trainer and enable it to generate controlled trajectories from other environments and random GP controls

LatentODE 
- test LatentODE with other controlled trajectories
- train with a larger dataset
- start implementing ANODE