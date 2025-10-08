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

"""Online Reinforcement Learning with user feedback for Food4All."""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque

from .reward_function import RewardFunction, FoodBankEntry, parse_output, RewardComponents


@dataclass
class UserFeedback:
    """Container for user feedback."""
    query: str
    output: str
    feedback_type: str  # 'pairwise' or 'questionnaire'
    timestamp: float

    # Pairwise preference fields
    preferred_output: Optional[str] = None
    rejected_output: Optional[str] = None

    # Questionnaire fields
    is_accurate: Optional[bool] = None
    incorrect_parts: Optional[List[str]] = None  # ['location', 'items', 'nutrition']
    free_text_comment: Optional[str] = None


@dataclass
class OnlineFeedbackBuffer:
    """Buffer for storing online feedback."""
    max_size: int = 1000
    feedback_history: deque = None

    def __post_init__(self):
        if self.feedback_history is None:
            self.feedback_history = deque(maxlen=self.max_size)

    def add(self, feedback: UserFeedback):
        """Add feedback to buffer."""
        self.feedback_history.append(feedback)

    def get_recent(self, n: int) -> List[UserFeedback]:
        """Get n most recent feedback entries."""
        return list(self.feedback_history)[-n:]

    def save(self, path: str):
        """Save buffer to disk."""
        data = [asdict(fb) for fb in self.feedback_history]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.feedback_history.clear()
        for item in data:
            self.feedback_history.append(UserFeedback(**item))


class OnlineRLLoop:
    """Online RL loop with user feedback."""

    def __init__(
        self,
        reward_function: RewardFunction,
        feedback_buffer: Optional[OnlineFeedbackBuffer] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        baseline_momentum: float = 0.9,
    ):
        """
        Initialize online RL loop.

        Args:
            reward_function: Multi-component reward function
            feedback_buffer: Buffer for storing feedback
            beta: Temperature for preference learning
            learning_rate: Learning rate for policy updates
            baseline_momentum: Momentum for baseline (variance reduction)
        """
        self.reward_function = reward_function
        self.feedback_buffer = feedback_buffer or OnlineFeedbackBuffer()
        self.beta = beta
        self.learning_rate = learning_rate

        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum

        # Adaptive reward weights (EMA)
        self.adaptive_weights = {
            'geo': 0.25,
            'items': 0.25,
            'nutr': 0.25,
            'hall': 0.25,
        }
        self.weight_momentum = 0.95

    def collect_pairwise_feedback(
        self,
        query: str,
        output_1: str,
        output_2: str,
        user_choice_fn: Optional[Callable[[str, str, str], int]] = None
    ) -> UserFeedback:
        """
        Collect pairwise preference feedback from user.

        Args:
            query: User query
            output_1: First candidate output
            output_2: Second candidate output
            user_choice_fn: Function that prompts user and returns choice (1 or 2)
                           If None, uses CLI input

        Returns:
            UserFeedback object
        """
        if user_choice_fn is None:
            # Default CLI prompt
            print("\n" + "="*80)
            print(f"QUERY: {query}")
            print("="*80)
            print(f"\nOPTION 1:\n{output_1}")
            print("\n" + "-"*80)
            print(f"\nOPTION 2:\n{output_2}")
            print("\n" + "-"*80)

            while True:
                choice = input("\nWhich output do you prefer? (1 or 2): ").strip()
                if choice in ['1', '2']:
                    choice = int(choice)
                    break
                print("Invalid input. Please enter 1 or 2.")
        else:
            choice = user_choice_fn(query, output_1, output_2)

        preferred = output_1 if choice == 1 else output_2
        rejected = output_2 if choice == 1 else output_1

        feedback = UserFeedback(
            query=query,
            output=preferred,
            feedback_type='pairwise',
            timestamp=time.time(),
            preferred_output=preferred,
            rejected_output=rejected,
        )

        self.feedback_buffer.add(feedback)
        return feedback

    def collect_questionnaire_feedback(
        self,
        query: str,
        output: str,
        questionnaire_fn: Optional[Callable[[str, str], Dict]] = None
    ) -> UserFeedback:
        """
        Collect questionnaire-based feedback from user.

        Args:
            query: User query
            output: Model output
            questionnaire_fn: Function that prompts user and returns responses
                             If None, uses CLI input

        Returns:
            UserFeedback object
        """
        if questionnaire_fn is None:
            # Default CLI questionnaire
            print("\n" + "="*80)
            print(f"QUERY: {query}")
            print("="*80)
            print(f"\nOUTPUT:\n{output}")
            print("\n" + "-"*80)

            # Question 1: Overall accuracy
            while True:
                accurate = input("\nWas the provided information accurate? (yes/no): ").strip().lower()
                if accurate in ['yes', 'no', 'y', 'n']:
                    is_accurate = accurate in ['yes', 'y']
                    break
                print("Invalid input. Please enter yes or no.")

            # Question 2: Incorrect parts (if not accurate)
            incorrect_parts = []
            if not is_accurate:
                print("\nWhich parts were incorrect?")
                print("1. Location/ZIP code")
                print("2. Food items")
                print("3. Nutritional information")
                print("4. Food bank names (hallucination)")
                parts_input = input("Enter numbers separated by commas (e.g., 1,3): ").strip()

                part_map = {
                    '1': 'location',
                    '2': 'items',
                    '3': 'nutrition',
                    '4': 'hallucination',
                }
                for num in parts_input.split(','):
                    num = num.strip()
                    if num in part_map:
                        incorrect_parts.append(part_map[num])

            # Question 3: Free text comment
            comment = input("\nAny additional comments? (optional): ").strip()
            comment = comment if comment else None

            responses = {
                'is_accurate': is_accurate,
                'incorrect_parts': incorrect_parts,
                'free_text_comment': comment,
            }
        else:
            responses = questionnaire_fn(query, output)

        feedback = UserFeedback(
            query=query,
            output=output,
            feedback_type='questionnaire',
            timestamp=time.time(),
            is_accurate=responses['is_accurate'],
            incorrect_parts=responses.get('incorrect_parts', []),
            free_text_comment=responses.get('free_text_comment'),
        )

        self.feedback_buffer.add(feedback)
        return feedback

    def compute_preference_loss(
        self,
        feedback: UserFeedback,
        zipcode: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute preference loss from pairwise feedback.

        L_pref = -log σ(β * (R(y* | q) - R(y_reject | q)))

        Args:
            feedback: UserFeedback with pairwise preference
            zipcode: Query ZIP code

        Returns:
            (loss, metrics) tuple
        """
        pred_preferred = parse_output(feedback.preferred_output)
        pred_rejected = parse_output(feedback.rejected_output)

        r_preferred, _ = self.reward_function.compute_reward(zipcode, pred_preferred)
        r_rejected, _ = self.reward_function.compute_reward(zipcode, pred_rejected)

        delta_r = r_preferred - r_rejected

        # Preference loss
        loss = -np.log(self._sigmoid(self.beta * delta_r) + 1e-8)

        metrics = {
            'loss': float(loss),
            'r_preferred': float(r_preferred),
            'r_rejected': float(r_rejected),
            'delta_r': float(delta_r),
        }

        return float(loss), metrics

    def compute_reward_shaping_from_questionnaire(
        self,
        feedback: UserFeedback
    ) -> RewardComponents:
        """
        Map questionnaire responses to reward components.

        Args:
            feedback: UserFeedback from questionnaire

        Returns:
            RewardComponents with shaped rewards
        """
        # Default neutral rewards
        r_geo = 0.5
        r_items = 0.5
        r_nutr = 0.5
        r_hall = 0.5

        if feedback.is_accurate:
            # All components get high reward
            r_geo = 1.0
            r_items = 1.0
            r_nutr = 1.0
            r_hall = 1.0
        else:
            # Penalize specific incorrect components
            if feedback.incorrect_parts:
                if 'location' in feedback.incorrect_parts:
                    r_geo = 0.0
                if 'items' in feedback.incorrect_parts:
                    r_items = 0.0
                if 'nutrition' in feedback.incorrect_parts:
                    r_nutr = 0.0
                if 'hallucination' in feedback.incorrect_parts:
                    r_hall = 0.0

        return RewardComponents(
            r_geo=r_geo,
            r_items=r_items,
            r_nutr=r_nutr,
            r_hall=r_hall,
        )

    def update_adaptive_weights(self, reward_components: RewardComponents):
        """
        Update adaptive reward weights based on user feedback.

        Args:
            reward_components: Reward components from feedback
        """
        # Components with low rewards should get higher weight (user cares more)
        # Use inverse weighting with EMA
        component_values = {
            'geo': reward_components.r_geo,
            'items': reward_components.r_items,
            'nutr': reward_components.r_nutr,
            'hall': reward_components.r_hall,
        }

        # Compute inverse weights (low reward -> high weight)
        inverse_weights = {k: 1.0 - v for k, v in component_values.items()}
        total_inverse = sum(inverse_weights.values())

        if total_inverse > 0:
            normalized_weights = {k: v / total_inverse for k, v in inverse_weights.items()}

            # EMA update
            for key in self.adaptive_weights:
                self.adaptive_weights[key] = (
                    self.weight_momentum * self.adaptive_weights[key] +
                    (1 - self.weight_momentum) * normalized_weights[key]
                )

        # Update reward function weights
        self.reward_function.weights = self.adaptive_weights.copy()

    def update_policy(
        self,
        feedback: UserFeedback,
        zipcode: str
    ) -> Dict[str, float]:
        """
        Update policy based on feedback.

        Args:
            feedback: User feedback
            zipcode: Query ZIP code

        Returns:
            Update metrics
        """
        if feedback.feedback_type == 'pairwise':
            # Pairwise preference update
            loss, metrics = self.compute_preference_loss(feedback, zipcode)

            # Update baseline
            self.baseline = (
                self.baseline_momentum * self.baseline +
                (1 - self.baseline_momentum) * metrics['r_preferred']
            )

            metrics['baseline'] = self.baseline
            return metrics

        elif feedback.feedback_type == 'questionnaire':
            # Questionnaire reward shaping
            reward_components = self.compute_reward_shaping_from_questionnaire(feedback)

            # Update adaptive weights
            self.update_adaptive_weights(reward_components)

            # Compute shaped reward
            r_online = reward_components.aggregate(self.adaptive_weights)

            # Update baseline
            self.baseline = (
                self.baseline_momentum * self.baseline +
                (1 - self.baseline_momentum) * r_online
            )

            metrics = {
                'r_online': float(r_online),
                'baseline': self.baseline,
                'r_geo': reward_components.r_geo,
                'r_items': reward_components.r_items,
                'r_nutr': reward_components.r_nutr,
                'r_hall': reward_components.r_hall,
                'weight_geo': self.adaptive_weights['geo'],
                'weight_items': self.adaptive_weights['items'],
                'weight_nutr': self.adaptive_weights['nutr'],
                'weight_hall': self.adaptive_weights['hall'],
            }

            return metrics

        return {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _extract_zipcode(query: str) -> str:
        """Extract ZIP code from query."""
        import re
        match = re.search(r'\b(\d{5})\b', query)
        return match.group(1) if match else "00000"

    def run_online_loop(
        self,
        query: str,
        generate_output_fn: Callable[[str], str],
        num_iterations: int = 1,
        feedback_mode: str = 'pairwise',  # 'pairwise' or 'questionnaire'
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Run online RL loop for a query.

        Args:
            query: User query
            generate_output_fn: Function that generates model output from query
            num_iterations: Number of feedback iterations
            feedback_mode: Type of feedback to collect
            verbose: Whether to print updates

        Returns:
            List of metrics for each iteration
        """
        metrics_history = []
        zipcode = self._extract_zipcode(query)

        for i in range(num_iterations):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Online RL Iteration {i+1}/{num_iterations}")
                print(f"{'='*80}")

            if feedback_mode == 'pairwise':
                # Generate two candidate outputs
                output_1 = generate_output_fn(query)
                output_2 = generate_output_fn(query)

                # Collect pairwise feedback
                feedback = self.collect_pairwise_feedback(query, output_1, output_2)
            else:
                # Generate single output
                output = generate_output_fn(query)

                # Collect questionnaire feedback
                feedback = self.collect_questionnaire_feedback(query, output)

            # Update policy
            metrics = self.update_policy(feedback, zipcode)
            metrics_history.append(metrics)

            if verbose:
                print(f"\nUpdate metrics: {json.dumps(metrics, indent=2)}")

        return metrics_history
