# Continuous time model-based reinforcement learning

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/eliottprdlx/CTMBRL
cd CTMBRL
```

---

### 2. Create a virtual environment

#### 🔵 On MacOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

#### 🔵 On Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

Make sure your virtual environment is **activated**, then install the required packages:

```bash
pip install -r requirements.txt
```

---

## How to use

1. Generate trajectories using traj_generator/main.py

2. Train a VAE model on these trajectories using latent_ode_vae/main.py