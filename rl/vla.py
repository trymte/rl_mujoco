"""Lightweight Vision-Language-Action model for Humanoid-v4.

This module provides a VLA architecture suitable for representation learning
on the Humanoid walking task. It uses SigLIP for vision-language encoding
and a lightweight action head for continuous control.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIPVLA(nn.Module):
    """Vision-Language-Action model using SigLIP for representation learning.

    Architecture:
    - SigLIP base (400M params) for joint vision-language encoding
    - Lightweight MLP action head for continuous control
    - Optional fine-tuning of vision encoder
    """

    def __init__(
        self,
        action_dim: int = 17,  # Humanoid-v4 action space
        hidden_dim: int = 512,
        freeze_vision_encoder: bool = True,
        model_name: str = "google/siglip-base-patch16-224",
    ):
        super().__init__()

        self.siglip = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        if freeze_vision_encoder:
            for param in self.siglip.parameters():
                param.requires_grad = False

        self.embed_dim = self.siglip.config.vision_config.hidden_size  # 768 for base

        self.action_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Bound actions to [-1, 1]
        )

    def encode_vision_language(
        self,
        images: torch.Tensor,
        text_prompts: list[str],
    ) -> torch.Tensor:
        """Encode images and text into joint embeddings.

        Args:
            images: Batch of images [B, C, H, W]
            text_prompts: List of text prompts (length B)

        Returns:
            Joint embeddings [B, embed_dim]
        """
        inputs = self.processor(
            text=text_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(images.device) for k, v in inputs.items()}

        outputs = self.siglip(**inputs)

        # Use vision embeddings as primary representation
        # (SigLIP is trained for vision-text alignment)
        vision_embeds = outputs.vision_model_output.pooler_output

        return vision_embeds

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: list[str],
    ) -> torch.Tensor:
        """Forward pass: images + text -> actions.

        Args:
            images: Batch of images [B, C, H, W]
            text_prompts: List of text prompts (length B)

        Returns:
            Actions [B, action_dim]
        """
        # Get vision-language representations
        embeddings = self.encode_vision_language(images, text_prompts)

        # Predict actions
        actions = self.action_head(embeddings)

        return actions

    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder for fine-tuning."""
        for param in self.siglip.parameters():
            param.requires_grad = True


class HumanoidVisionWrapper(gym.Wrapper):
    """Wrapper to add visual observations to Humanoid-v4.

    Renders RGB observations from MuJoCo camera and provides them
    alongside proprioceptive state.

    Note: The environment should be created with render_mode="rgb_array" for rendering.
    """

    def __init__(
        self,
        env: gym.Env,
        camera_id: int = 0,  # 0 = default tracking camera
        image_size: tuple[int, int] = (224, 224),
        include_state: bool = True,
    ):
        # Ensure the env has rgb_array rendering capability
        if not hasattr(env.unwrapped, "mujoco_renderer"):
            # Need to recreate with render_mode
            env_id = env.spec.id if hasattr(env, "spec") and env.spec else "Humanoid-v4"
            import warnings

            warnings.warn(
                f"Environment {env_id} doesn't have mujoco_renderer. "
                "Consider creating with render_mode='rgb_array'"
            )

        super().__init__(env)
        self.camera_id = camera_id
        self.image_size = image_size
        self.include_state = include_state

        # Update observation space
        if include_state:
            self.observation_space = gym.spaces.Dict(
                {
                    "pixels": gym.spaces.Box(0, 255, (*image_size, 3), dtype=np.uint8),
                    "state": env.observation_space,
                }
            )
        else:
            self.observation_space = gym.spaces.Box(
                0, 255, (*image_size, 3), dtype=np.uint8
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, state_obs):
        """Get observation with visual component."""
        pixels = self._render_observation()

        if self.include_state:
            return {"pixels": pixels, "state": state_obs}
        else:
            return pixels

    def _render_observation(self) -> np.ndarray:
        """Render RGB observation from a MuJoCo camera."""
        # Access the base MuJoCo environment
        base_env = self.env.unwrapped

        # Use MuJoCo's renderer to get RGB array
        # The camera_id parameter: -1 = default, or specific camera index
        if (
            hasattr(base_env, "mujoco_renderer")
            and base_env.mujoco_renderer is not None
        ):
            # Render using the renderer
            pixels = base_env.mujoco_renderer.render(render_mode="rgb_array")
        else:
            # Fallback: create a temporary renderer
            import mujoco

            renderer = mujoco.Renderer(base_env.model, height=480, width=640)
            renderer.update_scene(base_env.data, camera=self.camera_id)
            pixels = renderer.render()

        # Resize to target size
        img = Image.fromarray(pixels).resize(self.image_size)
        return np.array(img)


class VLAAgent:
    """Agent wrapper for VLA model with language-conditioned control.

    Provides a simple interface for:
    - Loading/saving models
    - Predicting actions given images and instructions
    - Training with imitation learning or RL
    """

    def __init__(
        self,
        model: SigLIPVLA,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_instruction: str = "walk forward smoothly",
    ):
        self.model = model.to(device)
        self.device = device
        self.default_instruction = default_instruction

    def predict(
        self,
        image: np.ndarray,
        instruction: str | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """Predict action from image and instruction.

        Args:
            image: RGB image [H, W, 3]
            instruction: Text instruction (uses default if None)
            deterministic: Whether to use deterministic policy (no noise)

        Returns:
            action: Action array [action_dim]
            state: None (for compatibility with SB3 interface)
        """
        if instruction is None:
            instruction = self.default_instruction

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Ensure correct format [B, C, H, W]
        if image.shape[-1] == 3:  # [B, H, W, C]
            image = image.permute(0, 3, 1, 2)

        image = image.to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            action = self.model(image, [instruction])

        return action.cpu().numpy()[0], None

    def save(self, path: str):
        """Save model weights."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "default_instruction": self.default_instruction,
            },
            path,
        )

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.default_instruction = checkpoint.get(
            "default_instruction", self.default_instruction
        )


def create_vla_agent(
    action_dim: int = 17,
    hidden_dim: int = 512,
    freeze_encoder: bool = True,
    device: str | None = None,
    model_name: str = "google/siglip-base-patch16-224",
) -> VLAAgent:
    """Factory function to create a VLA agent.

    Args:
        action_dim: Action space dimensionality
        hidden_dim: Hidden dimension for action head
        freeze_encoder: Whether to freeze vision encoder initially
        device: Device to use (auto-detect if None)
        model_name: HuggingFace model name for SigLIP

    Returns:
        VLAAgent ready for training or inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SigLIPVLA(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        freeze_vision_encoder=freeze_encoder,
        model_name=model_name,
    )

    return VLAAgent(model, device=device)


def train_vla_behavioral_cloning(
    vla_agent: VLAAgent,
    dataset_name: str = "humanoid/expert-v1",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    instructions: list[str] | None = None,
):
    """Train VLA using behavioral cloning on a Minari dataset.

    Args:
        vla_agent: VLA agent to train
        dataset_name: Minari dataset name
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        instructions: List of instructions to use (randomly sampled)
    """
    import minari
    from tqdm.auto import tqdm

    # Default instructions for humanoid walking
    if instructions is None:
        instructions = [
            "walk forward smoothly",
            "walk forward at steady pace",
            "move forward maintaining balance",
            "walk straight ahead",
            "advance forward carefully",
        ]

    minari_dataset = minari.load_dataset(dataset_name)
    env = minari_dataset.recover_environment()

    vision_env = HumanoidVisionWrapper(env)

    print(f"Training VLA on {len(minari_dataset)} episodes...")
    print(f"Action dim: {env.action_space.shape[0]}")

    # Setup training
    optimizer = torch.optim.AdamW(
        vla_agent.model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    loss_fn = nn.MSELoss()

    # Training loop
    vla_agent.model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Iterate through episodes
        for episode in tqdm(minari_dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get observations and actions
            observations = episode.observations  # State observations
            actions = episode.actions

            # Render visual observations for each timestep
            # (In practice, you'd want to pre-render and cache these)
            images = []
            env.reset()
            for obs in observations:
                env.unwrapped.set_state(obs[:27], obs[27:])  # Humanoid qpos/qvel
                img = vision_env._render_observation()
                images.append(img)

            # Create batches
            num_steps = len(actions)
            for i in range(0, num_steps, batch_size):
                batch_images = images[i : i + batch_size]
                batch_actions = actions[i : i + batch_size]

                if len(batch_images) == 0:
                    continue

                # Random instruction for this batch
                instruction = np.random.choice(instructions)
                batch_instructions = [instruction] * len(batch_images)

                # Convert to tensors
                batch_images = (
                    torch.from_numpy(np.stack(batch_images))
                    .float()
                    .permute(0, 3, 1, 2)
                    .to(vla_agent.device)
                )
                batch_actions = (
                    torch.from_numpy(batch_actions).float().to(vla_agent.device)
                )

                # Forward pass
                pred_actions = vla_agent.model(batch_images, batch_instructions)
                loss = loss_fn(pred_actions, batch_actions)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vla_agent.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")


def evaluate_vla(
    vla_agent: VLAAgent,
    env_id: str = "Humanoid-v4",
    num_episodes: int = 5,
    instruction: str = "walk forward smoothly",
    render: bool = False,
) -> dict:
    """Evaluate VLA agent on environment.

    Args:
        vla_agent: VLA agent to evaluate
        env_id: Environment ID
        num_episodes: Number of evaluation episodes
        instruction: Instruction for the agent
        render: Whether to render environment

    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment with rgb_array for vision wrapper
    # If render is True, we'll show it separately
    env = gym.make(env_id, render_mode="rgb_array")
    env = HumanoidVisionWrapper(env, include_state=False)

    # If user wants to see rendering, create a separate viewer env
    if render:
        import warnings

        warnings.warn(
            "Visual rendering not yet implemented in evaluation. Use env render_mode='human' separately."
        )

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done and episode_length < 1000:
            action, _ = vla_agent.predict(obs, instruction)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(
            f"Episode {ep+1}: Reward = {episode_reward:.2f}, Length = {episode_length}"
        )

    env.close()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards,
    }


if __name__ == "__main__":
    print("Creating VLA agent...")
    agent = create_vla_agent(
        action_dim=17,
        hidden_dim=512,
        freeze_encoder=True,
    )

    print(f"Model parameters: {sum(p.numel() for p in agent.model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in agent.model.parameters() if p.requires_grad):,}"
    )

    print("\nTesting forward pass...")
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_instruction = ["walk forward"]
    action = agent.model(dummy_image, dummy_instruction)
    print(f"Output action shape: {action.shape}")
    print(f"Action range: [{action.min().item():.2f}, {action.max().item():.2f}]")

    print("nVLA agent ready for training!")
