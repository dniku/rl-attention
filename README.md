# rl-attention

### Instructions

To use the launcher, run in the cloned repository:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

To use the launcher, just run `main.py`. All parameters are stored in `config.json`.

Adding a new model architecture is essentially replacing a `Policy`. Currently `config.json` specifies a `CnnPolicy`
which comes bundled with `stable-baselines`. See `stable_baselines/common/policies.py` for examples of how to define
custom policies.

We also include a copy of the code for training algorithms here so that it can be modified more easily.

The complete trained model is stored in stored under `saved_models` as `env_name-model_name-policy_type.pkl`.

These can then be loaded easily. Saving the metrics had a bug so I've cut that out and will sort it out soon, but since
the full model is saved this isn't super urgent.

### Jupyter instructions

In the virtual environment set up above ...

```bash
pip install jupyter
ipython kernel install --user --name=.env
jupyter notebook
```
