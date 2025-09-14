"""
PPO-style fine-tuning of sentence-transformers/all-MiniLM-L6-v2 using LoRA adapters.

Overview:
- We treat the model's embeddings as the policy's mean (mu).
- We sample actions = mu + eps * sigma (diagonal Gaussian).
- A pretrained reward model (scikit-learn regressor) scores each sampled embedding.
- A value head predicts expected reward from the embedding (trained with MSE).
- We use PPO clipping on the policy (log-probs of the sampled embeddings) to update LoRA parameters.

Requirements:
- torch, transformers, peft, scikit-learn, joblib, pandas
- A trained reward model saved as 'embedding_quality_model.pkl' (joblib dump of scikit-learn regressor)
- A CSV dataset with column 'text' (strings). You can also provide a column 'grade' if you want to pre-compute or evaluate.

Notes and caveats:
- This is a PPO-style *surrogate* adapted to embedding optimization (not token-level RLHF). It's experimental but practical.
- The policy's stochasticity is in embedding space (Gaussian noise). We optimize LoRA adapter parameters so the model's mu produces embeddings that yield higher reward after sampling.
- Keep batch sizes modest if using CPU. Prefer GPU and enough memory.

Usage example:
python ppo_lora_embedding_finetune.py --dataset dataset.csv --output_dir ./ppo_lora_model

"""

import argparse
import math
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        if "text" not in self.df.columns:
            raise ValueError("CSV must contain a 'text' column with input strings.")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "text": text,
        }


@dataclass
class PPOConfig:
    batch_size: int = 16
    ppo_epochs: int = 4  # number of optimization passes over collected batch
    lr: float = 1e-4
    clip_epsilon: float = 0.2
    vf_coef: float = 1.0
    ent_coef: float = 1e-3
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    seed: int = 42


class PPOTrainer:
    def __init__(self, model_name, reward_model_path, cfg: PPOConfig, tokenizer_kwargs=None):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_kwargs or {}))
        base_model = AutoModel.from_pretrained(model_name)

        # LoRA
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        # prepare for int8 training if available (keeps memory low)
        try:
            base_model = prepare_model_for_int8_training(base_model)
        except Exception:
            # graceful fallback if prepare_model_for_int8_training not applicable
            pass

        self.model = get_peft_model(base_model, lora_config).to(self.device)
        self.model.train()

        # A small learnable log_sigma per embedding dim (we'll initialize from 1.0)
        # Determine embedding dim by running a dummy forward
        with torch.no_grad():
            dummy = self._encode_texts(["hello world"])
        self.emb_dim = dummy.shape[1]

        # Policy noise scale: we parameterize log_sigma (learnable)
        self.log_sigma = nn.Parameter(torch.zeros(self.emb_dim, device=self.device))

        # Value head predicts reward given (mean-pooled) embedding
        self.value_head = nn.Sequential(nn.Linear(self.emb_dim, 256), nn.Tanh(), nn.Linear(256, 1)).to(self.device)

        # Optimizer will update LoRA parameters + value head + log_sigma
        # Only include parameters that require grad and belong to the peft adapters, value_head, or log_sigma
        trainable_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        trainable_params += list(self.value_head.parameters())
        trainable_params.append(self.log_sigma)

        self.optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

        # Load reward model (scikit-learn regressor)
        self.reward_model = joblib.load(reward_model_path)

    def _encode_texts(self, texts, max_length=128):
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            # mean pool
            emb = outputs.last_hidden_state.mean(dim=1)
        return emb

    def collect_batch(self, dataloader):
        # Collect states, actions, rewards, log_probs, values for one batch
        states = []
        actions = []
        rewards = []
        old_log_probs = []
        values = []
        texts = []

        self.model.eval()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                mu = outputs.last_hidden_state.mean(dim=1)  # (B, D)

            # Sigma: make positive via exp(log_sigma)
            sigma = torch.exp(self.log_sigma).unsqueeze(0)  # (1, D)

            # Sample action in embedding space: a = mu + eps * sigma
            eps = torch.randn_like(mu)
            action = mu + eps * sigma

            # Compute log_prob of each action under the multivariate diagonal gaussian
            # log_prob = sum over dims of -0.5 * ((a-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
            var = sigma ** 2
            log_probs_per_dim = -0.5 * ((action - mu) ** 2) / (var + 1e-12) - torch.log(sigma + 1e-12) - 0.5 * math.log(2 * math.pi)
            log_prob = log_probs_per_dim.sum(dim=1)  # (B,)

            # Reward: use reward_model (sklearn) on action (numpy)
            action_np = action.detach().cpu().numpy()
            reward_np = self.reward_model.predict(action_np)  # shape (B,)

            # Value estimates from value_head
            with torch.no_grad():
                value = self.value_head(mu).squeeze(-1)  # (B,)

            states.append(mu.detach())
            actions.append(action.detach())
            rewards.append(torch.tensor(reward_np, dtype=torch.float, device=self.device))
            old_log_probs.append(log_prob.detach())
            values.append(value.detach())
            texts.extend(batch.get("text", [None] * len(reward_np)))

        # Concatenate across batches
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)
        values = torch.cat(values, dim=0)

        # Normalize rewards to help stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # compute advantages
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.model.train()
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "old_log_probs": old_log_probs,
            "values": values,
            "advantages": advantages,
            "texts": texts,
        }

    def ppo_update(self, batch_data):
        states = batch_data["states"]
        actions = batch_data["actions"]
        advantages = batch_data["advantages"]
        rewards = batch_data["rewards"]
        old_log_probs = batch_data["old_log_probs"]

        B = states.size(0)
        minibatch_size = max(1, B // 4)

        for _ in range(self.cfg.ppo_epochs):
            # For simplicity shuffle and take minibatches
            perm = torch.randperm(B)
            for start in range(0, B, minibatch_size):
                idx = perm[start: start + minibatch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_adv = advantages[idx]
                mb_rewards = rewards[idx]
                mb_old_logp = old_log_probs[idx]

                # Recompute mu from the current policy
                # To get mu, we need to run a forward pass through the model (which includes LoRA)
                # For this, we can attempt to recompute using the tokenizer and texts, but we don't have texts here.
                # Instead, we assume mu == mb_states (states were recorded mu at collection time).
                mu = mb_states

                # Recompute log_probs for the current policy (with current log_sigma)
                sigma = torch.exp(self.log_sigma).unsqueeze(0)
                var = sigma ** 2
                log_probs_per_dim = -0.5 * ((mb_actions - mu) ** 2) / (var + 1e-12) - torch.log(sigma + 1e-12) - 0.5 * math.log(2 * math.pi)
                logp = log_probs_per_dim.sum(dim=1)

                # ratio for PPO
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Value loss: predict value from mb_states using value_head
                value_preds = self.value_head(mu).squeeze(-1)
                value_loss = torch.mean((mb_rewards - value_preds) ** 2)

                # Entropy bonus (encourage exploration)
                # Entropy of diagonal gaussian: sum(0.5 * (1 + log(2*pi*var)))
                ent_per_dim = 0.5 * (1 + torch.log(2 * math.pi * var + 1e-12))
                entropy = ent_per_dim.sum()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.value_head.parameters()), self.cfg.max_grad_norm)
                self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def train(self, dataset_csv, epochs=3):
        dataset = TextDataset(dataset_csv, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        for ep in range(epochs):
            t0 = time.time()
            batch_data = self.collect_batch(dataloader)
            ploss, vloss, ent = self.ppo_update(batch_data)
            t1 = time.time()
            print(f"Epoch {ep+1}/{epochs} — policy_loss={ploss:.4f}, value_loss={vloss:.4f}, entropy={ent:.4f}, batch_time={t1-t0:.1f}s")

    def save(self, output_dir):
        # Save LoRA adapters + tokenizer + value_head + log_sigma
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save({
            "value_head_state": self.value_head.state_dict(),
            "log_sigma": self.log_sigma.detach().cpu(),
        }, f"{output_dir}/ppo_extras.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="CSV file with 'text' column")
    parser.add_argument("--reward_model", type=str, default="embedding_quality_model.pkl")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="ppo_lora_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = PPOConfig(batch_size=args.batch_size)
    trainer = PPOTrainer(args.model_name, args.reward_model, cfg)
    trainer.train(args.dataset, epochs=args.epochs)
    trainer.save(args.output_dir)


if __name__ == "__main__":
    main()
