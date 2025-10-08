# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

"""Offline Reinforcement Learning with Direct Preference Optimization (DPO) for Food4All."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .reward_function import RewardFunction, FoodBankEntry, parse_output


@dataclass
class PreferenceDataset:
    """Dataset for offline preference learning."""
    queries: List[str]
    positive_outputs: List[str]
    negative_outputs: List[str]
    ground_truths: List[List[FoodBankEntry]]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx: int):
        return (
            self.queries[idx],
            self.positive_outputs[idx],
            self.negative_outputs[idx],
            self.ground_truths[idx]
        )


class OfflineRLTrainer:
    """Offline RL trainer using Direct Preference Optimization."""

    def __init__(
        self,
        reward_function: RewardFunction,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
    ):
        """
        Initialize offline RL trainer.

        Args:
            reward_function: Multi-component reward function
            beta: Temperature parameter for DPO
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
        """
        self.reward_function = reward_function
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))

    def compute_dpo_loss(
        self,
        query_zipcode: str,
        y_positive: str,
        y_negative: str,
        ground_truth: List[FoodBankEntry]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute DPO loss for a single example.

        L_DPO = -log σ(β * (R(y+ | q) - R(y- | q)))

        Args:
            query_zipcode: User query ZIP code
            y_positive: Preferred output
            y_negative: Rejected output
            ground_truth: Ground truth food banks

        Returns:
            (loss, metrics) tuple
        """
        # Parse outputs
        pred_positive = parse_output(y_positive)
        pred_negative = parse_output(y_negative)

        # Compute rewards
        r_positive, components_pos = self.reward_function.compute_reward(
            query_zipcode, pred_positive, ground_truth
        )
        r_negative, components_neg = self.reward_function.compute_reward(
            query_zipcode, pred_negative, ground_truth
        )

        # Compute preference margin
        delta_r = r_positive - r_negative

        # DPO loss
        loss = -np.log(self.sigmoid(self.beta * delta_r) + 1e-8)

        metrics = {
            'loss': float(loss),
            'r_positive': float(r_positive),
            'r_negative': float(r_negative),
            'delta_r': float(delta_r),
            'r_geo_pos': float(components_pos.r_geo),
            'r_items_pos': float(components_pos.r_items),
            'r_nutr_pos': float(components_pos.r_nutr),
            'r_hall_pos': float(components_pos.r_hall),
        }

        return float(loss), metrics

    def train_epoch(
        self,
        dataset: PreferenceDataset,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train one epoch on the preference dataset.

        Args:
            dataset: Preference dataset
            verbose: Whether to show progress bar

        Returns:
            Dictionary of epoch metrics
        """
        total_loss = 0.0
        total_delta_r = 0.0
        total_r_pos = 0.0
        total_r_neg = 0.0

        iterator = tqdm(range(len(dataset)), desc="Training") if verbose else range(len(dataset))

        for idx in iterator:
            query, y_pos, y_neg, ground_truth = dataset[idx]

            # Extract ZIP code from query
            zipcode = self._extract_zipcode(query)

            # Compute loss
            loss, metrics = self.compute_dpo_loss(zipcode, y_pos, y_neg, ground_truth)

            total_loss += loss
            total_delta_r += metrics['delta_r']
            total_r_pos += metrics['r_positive']
            total_r_neg += metrics['r_negative']

        # Aggregate metrics
        n = len(dataset)
        epoch_metrics = {
            'avg_loss': total_loss / n,
            'avg_delta_r': total_delta_r / n,
            'avg_r_positive': total_r_pos / n,
            'avg_r_negative': total_r_neg / n,
        }

        return epoch_metrics

    @staticmethod
    def _extract_zipcode(query: str) -> str:
        """Extract ZIP code from query text."""
        import re
        match = re.search(r'\b(\d{5})\b', query)
        return match.group(1) if match else "00000"

    def evaluate(
        self,
        dataset: PreferenceDataset,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on preference dataset.

        Args:
            dataset: Preference dataset
            verbose: Whether to show progress

        Returns:
            Evaluation metrics
        """
        correct_preferences = 0
        total = len(dataset)

        for idx in range(total):
            query, y_pos, y_neg, ground_truth = dataset[idx]
            zipcode = self._extract_zipcode(query)

            pred_pos = parse_output(y_pos)
            pred_neg = parse_output(y_neg)

            r_pos, _ = self.reward_function.compute_reward(zipcode, pred_pos, ground_truth)
            r_neg, _ = self.reward_function.compute_reward(zipcode, pred_neg, ground_truth)

            if r_pos > r_neg:
                correct_preferences += 1

        accuracy = correct_preferences / total

        return {
            'preference_accuracy': accuracy,
            'correct_preferences': correct_preferences,
            'total': total,
        }

    @staticmethod
    def load_dataset(data_path: str) -> PreferenceDataset:
        """
        Load preference dataset from JSON file.

        Expected format:
        [
            {
                "query": "I live in 46637...",
                "positive": "Food Bank A, 46637: ...",
                "negative": "Food Bank X, 99999: ...",
                "ground_truth": [
                    {
                        "name": "Food Bank A",
                        "zipcode": "46637",
                        "items": ["bread", "milk"],
                        "nutritional_info": {"bread": {"calories": 100, ...}}
                    }
                ]
            }
        ]

        Args:
            data_path: Path to JSON dataset

        Returns:
            PreferenceDataset object
        """
        with open(data_path, 'r') as f:
            data = json.load(f)

        queries = []
        positive_outputs = []
        negative_outputs = []
        ground_truths = []

        for example in data:
            queries.append(example['query'])
            positive_outputs.append(example['positive'])
            negative_outputs.append(example['negative'])

            # Parse ground truth
            gt_banks = []
            for bank_data in example.get('ground_truth', []):
                bank = FoodBankEntry(
                    name=bank_data['name'],
                    zipcode=bank_data['zipcode'],
                    items=bank_data.get('items', []),
                    nutritional_info=bank_data.get('nutritional_info', {}),
                    latitude=bank_data.get('latitude'),
                    longitude=bank_data.get('longitude'),
                )
                gt_banks.append(bank)

            ground_truths.append(gt_banks)

        return PreferenceDataset(queries, positive_outputs, negative_outputs, ground_truths)

    def save_checkpoint(self, path: str, metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'metrics': metrics,
            'beta': self.beta,
            'learning_rate': self.learning_rate,
        }
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)


def generate_negative_samples(
    positive_output: str,
    corruption_types: List[str] = ['zip_perturb', 'item_delete', 'nutrition_noise']
) -> str:
    """
    Generate negative samples from positive outputs via corruption.

    Args:
        positive_output: Ground truth / verified output
        corruption_types: Types of corruption to apply

    Returns:
        Corrupted negative output
    """
    import random
    import re

    output = positive_output

    if 'zip_perturb' in corruption_types:
        # Randomly perturb ZIP codes
        def perturb_zip(match):
            zipcode = match.group(1)
            perturbed = str(int(zipcode) + random.randint(-5000, 5000)).zfill(5)
            return perturbed

        output = re.sub(r'\b(\d{5})\b', perturb_zip, output)

    if 'item_delete' in corruption_types:
        # Delete random items
        items = output.split(';')
        if len(items) > 1:
            num_to_delete = max(1, len(items) // 3)
            indices_to_delete = random.sample(range(len(items)), num_to_delete)
            items = [item for i, item in enumerate(items) if i not in indices_to_delete]
            output = ';'.join(items)

    if 'nutrition_noise' in corruption_types:
        # Add noise to nutritional values
        def add_noise(match):
            value = float(match.group(1))
            noise = value * random.uniform(-0.3, 0.3)
            return f"{value + noise:.1f}"

        output = re.sub(r'(\d+\.?\d*)', add_noise, output)

    return output
