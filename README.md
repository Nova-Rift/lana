# Lana — our experimental PyTorch agent

Our existing custom agent, extracted into its own repository. The original
[`model.py`](survival_agent/hybrid/model.py) is preserved **byte for byte** from
Nova-Rift/survival-agent, commit `4b54ed9b08ad0d4bd9e23126ed85421802b48dbb`.
See [provenance](PROVENANCE.json). No third-party model implementation is included.

## Architecture

- **Causal Transformer:** memory over observations/actions, not a text LLM.
- **Categorical world model:** posterior/prior latents and observation, reward,
  continuation predictions.
- **Actor/value learning:** imagined rollouts plus TD learning.
- **Planning:** discrete MCTS or continuous MPC; planners never execute real actions.
- **Replay/checkpoints:** sequence replay and safe episode-boundary persistence.
- **Continual learning/forgetting:** no demonstrated retention or implemented
  neuron recycling. An ongoing learning loop alone is not evidence of these abilities.

## Run

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m survival_agent.hybrid.model --task CartPole-v1 --cycles 2 --output runs/smoke
```

The preserved CLI is a bounded CPU engineering example. The model is PyTorch and
can be placed on CUDA by a separate training adapter; this extraction deliberately
does not change the original file or silently migrate old checkpoints. No paid
compute, pretrained weights or benchmark claims are included.

Read `model.py` top to bottom: architecture → distributions → planners → replay →
losses → controller → training loop. `contracts.py` and `checkpoint_io.py` are
our own environment and serialization helpers, not additional neural models.

## Status

This repository separates source ownership and readability from earlier website
experiments. It does not make old arena checkpoints compatible with newly written
MuZero or DreamerV3. Existing published results remain historical evidence.
