# rl-attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pastafarianist/rl-attention/blob/master/attention_model.ipynb)

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
The config file and 100-step reward averages are stored under `saved_metrics` as `env_name-model_name-policy_type.txt`.

### Jupyter instructions

To install Jupyter, register a new kernel, and start a notebook, run in the virtual environment:
```bash
pip install jupyter
ipython kernel install --user --name=.env
jupyter notebook
```
Then activate the `.env` kernel in the notebook.

To make the logging work, add and execute after importing the `logging` module in the notebook:
```python
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
```
