from typing import Any

import optuna
import torch as th
from stable_baselines3.common.utils import LinearSchedule


def convert_onpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    hyperparams = sampled_params.copy()
    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    hyperparams["gae_lambda"] = 1 - sampled_params["one_minus_gae_lambda"]
    del hyperparams["one_minus_gae_lambda"]

    net_arch = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size", "n_steps"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
        "big": dict(pi=[400, 300], vf=[400, 300]),
        "large": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    }[net_arch]

    activation_fn_name = sampled_params["activation_fn"]
    del hyperparams["activation_fn"]

    activation_fn = {
        "tanh": th.nn.Tanh,
        "relu": th.nn.ReLU,
        "elu": th.nn.ELU,
        "leaky_relu": th.nn.LeakyReLU,
    }[activation_fn_name]

    # Handle learning rate schedule if present
    if "learning_rate_schedule" in hyperparams:
        if hyperparams["learning_rate_schedule"] == "linear":
            learning_rate = LinearSchedule(
                start=hyperparams["learning_rate"],
                end=hyperparams["learning_rate"] * 0.25,
                end_fraction=0.9,
            )
        else:
            learning_rate = hyperparams["learning_rate"]
        # Remove schedule from hyperparams (not a model parameter)
        del hyperparams["learning_rate_schedule"]
        hyperparams["learning_rate"] = learning_rate

    # Handle clip_range schedule if present (for PPO)
    if "clip_range_schedule" in hyperparams:
        if hyperparams["clip_range_schedule"] == "linear":
            clip_range = LinearSchedule(
                start=hyperparams["clip_range"],
                end=hyperparams["clip_range"] * 0.25,
                end_fraction=0.9,
            )
        else:
            clip_range = hyperparams["clip_range"]
        # Remove schedule from hyperparams
        del hyperparams["clip_range_schedule"]
        hyperparams["clip_range"] = clip_range

    return {
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
        **hyperparams,
    }


def convert_offpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    hyperparams = sampled_params.copy()

    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    net_arch = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "large": [256, 256, 256],
        "verybig": [512, 512, 512],
    }[net_arch]

    if "train_freq" in sampled_params:
        # Update to data ratio of 1, for n_envs=1
        hyperparams["gradient_steps"] = sampled_params["train_freq"]

        if "subsample_steps" in sampled_params:
            hyperparams["gradient_steps"] = max(
                sampled_params["train_freq"] // sampled_params["subsample_steps"], 1
            )
            del hyperparams["subsample_steps"]

    hyperparams["policy_kwargs"] = hyperparams.get("policy_kwargs", {})
    hyperparams["policy_kwargs"]["net_arch"] = net_arch

    if "activation_fn" in sampled_params:
        activation_fn_name = sampled_params["activation_fn"]
        del hyperparams["activation_fn"]

        activation_fn = {
            "tanh": th.nn.Tanh,
            "relu": th.nn.ReLU,
            "elu": th.nn.ELU,
            "leaky_relu": th.nn.LeakyReLU,
        }[activation_fn_name]
        hyperparams["policy_kwargs"]["activation_fn"] = activation_fn

    # Handle learning rate schedule if present
    if "learning_rate_schedule" in hyperparams:
        if hyperparams["learning_rate_schedule"] == "linear":
            learning_rate = LinearSchedule(
                start=hyperparams["learning_rate"],
                end=hyperparams["learning_rate"] * 0.25,
                end_fraction=0.9,
            )
        else:
            learning_rate = hyperparams["learning_rate"]
        # Remove schedule from hyperparams (not a model parameter)
        del hyperparams["learning_rate_schedule"]
        hyperparams["learning_rate"] = learning_rate

    # TQC/QRDQN
    if "n_quantiles" in sampled_params:
        del hyperparams["n_quantiles"]
        hyperparams["policy_kwargs"].update(
            {"n_quantiles": sampled_params["n_quantiles"]}
        )

    return hyperparams


def sample_ppo_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Sample core PPO hyperparameters from an Optuna trial."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    learning_rate_schedule = trial.suggest_categorical(
        "learning_rate_schedule", ["constant", "linear"]
    )
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    net_arch = trial.suggest_categorical(
        "net_arch", ["small", "medium", "big", "large"]
    )

    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10, 15])

    if batch_size > n_steps:
        batch_size = n_steps

    # Use one_minus_gamma for better log-scale sampling near 1
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.001, 0.1, log=True)
    one_minus_gae_lambda = trial.suggest_float(
        "one_minus_gae_lambda", 0.001, 0.2, log=True
    )

    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    clip_range_schedule = trial.suggest_categorical(
        "clip_range_schedule", ["constant", "linear"]
    )

    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.0001, 0.001, 0.002, 0.01])
    vf_coef = trial.suggest_categorical("vf_coef", [0.1, 0.25, 0.5, 0.75, 1.0])

    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0])

    params = {
        "learning_rate": learning_rate,
        "learning_rate_schedule": learning_rate_schedule,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "one_minus_gamma": one_minus_gamma,
        "one_minus_gae_lambda": one_minus_gae_lambda,
        "clip_range": clip_range,
        "clip_range_schedule": clip_range_schedule,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "activation_fn": activation_fn,
        "net_arch": net_arch,
    }
    return convert_onpolicy_params(params)


def sample_sac_params(
    trial: optuna.Trial,
    additional_args: dict | None = None,
) -> dict[str, Any]:
    """Sample SAC hyperparams from an Optuna trial.

    Args:
        trial: Optuna trial object
        n_actions: Number of actions (unused in basic SAC)
        n_envs: Number of parallel environments (default: 1)
        additional_args: Additional arguments for HER or TQC variants

    Returns:
        Dictionary of SAC hyperparameters
    """
    if additional_args is None:
        additional_args = {"using_her_replay_buffer": False}
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    # From 2**5=32 to 2**11=2048
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    # Polyak coeff
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)

    net_arch = trial.suggest_categorical(
        "net_arch", ["small", "medium", "big", "verybig"]
    )
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "net_arch": net_arch,
    }

    if additional_args["using_her_replay_buffer"]:
        hyperparams = sample_her_params(
            trial, hyperparams, additional_args["her_kwargs"]
        )

    if "sample_tqc" in additional_args:
        n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
        top_quantiles_to_drop_per_net = trial.suggest_int(
            "top_quantiles_to_drop_per_net", 0, min(n_quantiles - 1, 5)
        )
        hyperparams.update(
            {
                "n_quantiles": n_quantiles,
                "top_quantiles_to_drop_per_net": top_quantiles_to_drop_per_net,
            }
        )

    return convert_offpolicy_params(hyperparams)


def sample_her_params(
    trial: optuna.Trial, hyperparams: dict[str, Any], her_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Sample HerReplayBuffer hyperparams from an Optuna trial."""
    her_kwargs = her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams
