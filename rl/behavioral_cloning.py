# %% Behavioral cloning with PyTorch
import os
import sys

import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from minari import DataCollector
from rl_zoo3.train import train
from stable_baselines3 import SAC
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.manual_seed(42)

# %%
# Policy training
# ~~~~~~~~~~~~~~~~~~~
# Now we can train the expert policy using RL Baselines3 Zoo.
# We train a SAC agent on the Humanoid environment:

sys.argv = ["python", "--algo", "sac", "--env", "Humanoid-v4"]
train()
# This will generate a new folder named `log` with the expert policy.


# %%
# Dataset generation
# ~~~~~~~~~~~~~~~~~~~


env = DataCollector(gym.make("Humanoid-v4"))
path = os.path.abspath("") + "/logs/sac/Humanoid-v4_1/best_model"
agent = SAC.load(path)

total_episodes = 100  # Reduced for Humanoid due to longer episodes
for i in tqdm(range(total_episodes)):
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="humanoid/expert-v1",
    description="Expert policy for Humanoid-v4",
    algorithm_name="SAC",
    author="Trym",
    author_email="tengesdal1994@gmail.com",
)
# Once executing the script, the dataset will be saved on your disk. You can display the list of datasets with ``minari list local`` command.

# %%
# Behavioral cloning with PyTorch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Tanh to bound actions to [-1, 1]
        return x


# %%
# For the Humanoid environment, the output dimension is 17 (continuous action space), and the input dimension is 376 (observation space).
# Our next step is to load the dataset and set up the training loop. The ``MinariDataset`` is compatible with the PyTorch Dataset API, allowing us to load it directly using `PyTorch DataLoader <https://pytorch.org/docs/stable/data.html>`_.
# However, since each episode can have a varying length, we need to pad them.
# To achieve this, we can utilize the `collate_fn <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`_ feature of PyTorch DataLoader. Let's create the ``collate_fn`` function:


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch], batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch], batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


# %%
# We can now proceed to load the data and create the training loop.
# To begin, let's initialize the DataLoader, neural network, optimizer, and loss.


minari_dataset = minari.load_dataset("humanoid/expert-v1")
dataloader = DataLoader(
    minari_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
)

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space
assert isinstance(observation_space, spaces.Box)
assert isinstance(action_space, spaces.Box)  # Continuous action space

policy_net = PolicyNetwork(
    np.prod(observation_space.shape), np.prod(action_space.shape)
)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()  # MSE loss for continuous action regression

# %%
# We use MSE loss for regression, as the action space is continuous.
# We then train the policy to predict the actions:

num_epochs = 50

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        # Flatten batch dimensions for continuous actions
        observations = (
            batch["observations"][:, :-1]
            .reshape(-1, observation_space.shape[0])
            .float()
        )
        actions = batch["actions"].reshape(-1, action_space.shape[0]).float()

        # Remove padding (where all values are 0)
        mask = observations.abs().sum(dim=1) > 0
        observations = observations[mask]
        actions = actions[mask]

        if len(observations) == 0:
            continue

        a_pred = policy_net(observations)
        loss = loss_fn(a_pred, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    if num_batches > 0:
        print(f"Epoch: {epoch}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}")

# %%
# And now, we can evaluate if the policy learned from the expert!

env = gym.make("Humanoid-v4", render_mode="human")
obs, _ = env.reset(seed=42)
done = False
accumulated_rew = 0
step_count = 0
max_steps = 1000

while not done and step_count < max_steps:
    with torch.no_grad():
        action = policy_net(torch.Tensor(obs)).numpy()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    accumulated_rew += reward
    step_count += 1

env.close()
print(f"Accumulated reward over {step_count} steps: {accumulated_rew:.2f}")

# %%
# We can visually observe how well the learned policy performs on the Humanoid task.
#
