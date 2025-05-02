# 25/04/2025

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

# 28/04/2025

LatentODE 
- clean the code : ok
- add identity and laplace encoders : ok

# 29/04/2025

LatentODE 
- identify why performance is terrible : ok
- clean and modularize the LatentODE code : ok
- add interpolation for continuous control : ok
- add visualization : ok
- improved controlled trajectories generation : ok

# 30/04/2025

AC 
- remove : ok

latent_ode_vae
- clean and refactor code : ok
- modularize : ok

traj_generator
- create : ok

# 02/05/2025

latent_ode_vae
- added augmented diff eq solver based on "Augmented Neural ODEs" paper
- added length scheduler to train the model on length-increasing sub trajectories
- added adjoint_odeint compatibility
- major debugging

# TODO 

latent_ode_vae
- create a workflow to launch a testing phase
