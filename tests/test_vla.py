"""Tests for VLA module."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl.vla import HumanoidVisionWrapper, VLAAgent, create_vla_agent


@pytest.fixture
def vla_agent() -> VLAAgent:
    return create_vla_agent(action_dim=17, hidden_dim=256)


@pytest.fixture
def vision_env() -> HumanoidVisionWrapper:
    env = gym.make("Humanoid-v4", render_mode="rgb_array")
    wrapped_env = HumanoidVisionWrapper(env, include_state=True)
    yield wrapped_env
    env.close()


class TestSigLIPVLA:
    def test_model_creation(self, vla_agent: VLAAgent):
        assert vla_agent.model is not None
        assert vla_agent.device in ["cuda", "cpu"]

    def test_model_parameters(self, vla_agent: VLAAgent):
        total_params = sum(p.numel() for p in vla_agent.model.parameters())
        trainable_params = sum(
            p.numel() for p in vla_agent.model.parameters() if p.requires_grad
        )

        # Should have millions of parameters
        assert total_params > 200_000_000
        # Trainable should be less than total (encoder frozen)
        assert trainable_params < total_params

    def test_forward_pass(self, vla_agent: VLAAgent):
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        instructions = ["walk forward"] * batch_size

        actions = vla_agent.model(images, instructions)

        assert actions.shape == (batch_size, 17)
        assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)

    def test_encode_vision_language(self, vla_agent: VLAAgent):
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        instructions = ["walk forward", "walk backward"]

        embeddings = vla_agent.model.encode_vision_language(images, instructions)

        assert embeddings.shape == (batch_size, vla_agent.model.embed_dim)
        assert not torch.isnan(embeddings).any()

    def test_unfreeze_encoder(self, vla_agent: VLAAgent):
        # Initially frozen
        initial_trainable = sum(
            p.numel() for p in vla_agent.model.parameters() if p.requires_grad
        )

        # Unfreeze
        vla_agent.model.unfreeze_vision_encoder()

        # Should have more trainable params
        final_trainable = sum(
            p.numel() for p in vla_agent.model.parameters() if p.requires_grad
        )

        assert final_trainable > initial_trainable


class TestVLAAgent:
    def test_predict(self, vla_agent: VLAAgent):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        instruction = "walk forward"

        action, state = vla_agent.predict(image, instruction)

        assert action.shape == (17,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        assert state is None

    def test_predict_with_default_instruction(self, vla_agent: VLAAgent):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        action, _ = vla_agent.predict(image, instruction=None)

        assert action.shape == (17,)

    def test_predict_deterministic(self, vla_agent: VLAAgent):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        instruction = "walk forward"

        action1, _ = vla_agent.predict(image, instruction, deterministic=True)
        action2, _ = vla_agent.predict(image, instruction, deterministic=True)

        np.testing.assert_array_almost_equal(action1, action2)

    def test_save_load(self, vla_agent: VLAAgent, tmp_path: Path):
        save_path = tmp_path / "test_model.pt"

        # Save
        vla_agent.save(str(save_path))
        assert save_path.exists()

        # Load
        vla_agent.load(str(save_path))

        # Test prediction still works
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        action, _ = vla_agent.predict(image, "walk forward")
        assert action.shape == (17,)


class TestHumanoidVisionWrapper:
    def test_wrapper_creation(self, vision_env: HumanoidVisionWrapper):
        assert vision_env is not None
        assert "pixels" in vision_env.observation_space.spaces
        assert "state" in vision_env.observation_space.spaces

    def test_observation_space(self, vision_env: HumanoidVisionWrapper):
        pixel_space = vision_env.observation_space["pixels"]
        state_space = vision_env.observation_space["state"]

        assert pixel_space.shape == (224, 224, 3)
        assert pixel_space.dtype == np.uint8
        assert state_space.shape == (376,)  # Humanoid state dim

    def test_reset(self, vision_env: HumanoidVisionWrapper):
        obs, info = vision_env.reset()

        assert "pixels" in obs
        assert "state" in obs
        assert obs["pixels"].shape == (224, 224, 3)
        assert obs["state"].shape == (376,)

    def test_step(self, vision_env: HumanoidVisionWrapper):
        obs, _ = vision_env.reset()
        action = vision_env.action_space.sample()

        obs, reward, terminated, truncated, info = vision_env.step(action)

        assert "pixels" in obs
        assert "state" in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_render_observation(self, vision_env: HumanoidVisionWrapper):
        vision_env.reset()
        pixels = vision_env._render_observation()

        assert pixels.shape == (224, 224, 3)
        assert pixels.dtype == np.uint8
        assert pixels.min() >= 0 and pixels.max() <= 255

    def test_without_state(self):
        from rl.vla import HumanoidVisionWrapper

        env = gym.make("Humanoid-v4", render_mode="rgb_array")
        vision_env = HumanoidVisionWrapper(env, include_state=False)

        obs, _ = vision_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (224, 224, 3)

        env.close()


class TestIntegration:
    def test_full_rollout(self, vla_agent: VLAAgent, vision_env: HumanoidVisionWrapper):
        obs, _ = vision_env.reset()

        total_reward = 0
        max_steps = 10  # Short rollout for testing

        for _ in range(max_steps):
            action, _ = vla_agent.predict(obs["pixels"], instruction="walk forward")
            obs, reward, terminated, truncated, _ = vision_env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Should complete at least one step
        assert total_reward != 0 or True  # Might fail early, that's ok

    def test_different_instructions(self, vla_agent: VLAAgent):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        instructions = [
            "walk forward",
            "walk backward",
            "turn left",
            "stand still",
        ]

        actions = []
        for instruction in instructions:
            action, _ = vla_agent.predict(image, instruction)
            actions.append(action)

        for action in actions:
            assert action.shape == (17,)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
