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

"""Multi-component reward function for Food4All offline and online RL."""

import re
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt


@dataclass
class FoodBankEntry:
    """Represents a food bank with location and items."""
    name: str
    zipcode: str
    items: List[str]
    nutritional_info: Dict[str, Dict[str, float]]  # item -> {calories, protein, fat, carbs}
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class RewardComponents:
    """Individual reward components."""
    r_geo: float  # Geographic accuracy reward
    r_items: float  # Item coverage reward
    r_nutr: float  # Nutritional accuracy reward
    r_hall: float  # Hallucination penalty

    def aggregate(self, weights: Dict[str, float]) -> float:
        """Compute weighted sum of reward components."""
        return (
            weights.get('geo', 0.3) * self.r_geo +
            weights.get('items', 0.3) * self.r_items +
            weights.get('nutr', 0.3) * self.r_nutr +
            weights.get('hall', 0.1) * self.r_hall
        )


class RewardFunction:
    """Multi-component reward function for Food4All."""

    def __init__(
        self,
        verified_registry: Optional[Dict[str, FoodBankEntry]] = None,
        usda_database: Optional[Dict[str, Dict[str, float]]] = None,
        weights: Optional[Dict[str, float]] = None,
        lambda_hall: float = 0.4,
    ):
        """
        Initialize reward function.

        Args:
            verified_registry: Dict mapping food bank names to verified entries
            usda_database: Dict mapping food items to nutritional info
            weights: Component weights (geo, items, nutr, hall)
            lambda_hall: Penalty weight for hallucinated items
        """
        self.verified_registry = verified_registry or {}
        self.usda_database = usda_database or {}
        # Paper Table 2: (w1, w2, w3, w4) = (0.3, 0.3, 0.3, 0.1)
        self.weights = weights or {'geo': 0.3, 'items': 0.3, 'nutr': 0.3, 'hall': 0.1}
        self.lambda_hall = lambda_hall

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points in kilometers.

        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers
        r = 6371
        return c * r

    @staticmethod
    def zipcode_to_coords(zipcode: str) -> Tuple[float, float]:
        """
        Convert ZIP code to approximate coordinates (simplified version).
        In production, use a proper geocoding service.

        Args:
            zipcode: 5-digit ZIP code

        Returns:
            (latitude, longitude) tuple
        """
        # This is a placeholder - in production use real geocoding
        # For now, return approximate center of US
        zip_int = int(zipcode) if zipcode.isdigit() else 0
        # Simple heuristic mapping (NOT accurate, for demo only)
        lat = 35.0 + (zip_int % 1000) / 100.0
        lon = -95.0 - (zip_int // 1000) / 10.0
        return lat, lon

    def compute_geo_reward(
        self,
        target_zipcode: str,
        predicted_banks: List[FoodBankEntry]
    ) -> float:
        """
        Compute geographic accuracy reward.
        Rewards closer food banks, penalizes distant ones.

        Args:
            target_zipcode: User's target ZIP code
            predicted_banks: List of predicted food bank entries

        Returns:
            Geographic reward (higher is better, normalized to [0, 1])
        """
        if not predicted_banks:
            return 0.0

        target_lat, target_lon = self.zipcode_to_coords(target_zipcode)

        distances = []
        for bank in predicted_banks:
            if bank.latitude and bank.longitude:
                dist = self.haversine_distance(
                    target_lat, target_lon, bank.latitude, bank.longitude
                )
            else:
                # Fallback to ZIP code distance
                bank_lat, bank_lon = self.zipcode_to_coords(bank.zipcode)
                dist = self.haversine_distance(
                    target_lat, target_lon, bank_lat, bank_lon
                )
            distances.append(dist)

        # Geographic penalty in [-1, 0]: closer = less penalty (paper Appendix C)
        # r_geo = exp(-d/50) - 1; at d=0: 0, at d->inf: -1
        avg_distance = np.mean(distances)
        reward = np.exp(-avg_distance / 50.0) - 1.0
        return float(reward)

    def compute_items_reward(
        self,
        true_items: Set[str],
        predicted_items: Set[str]
    ) -> float:
        """
        Compute item coverage reward with hallucination penalty.

        r_items = |I_pred ∩ I_true| / |I_true| - λ * |I_pred \ I_true| / |I_pred|

        Args:
            true_items: Ground truth item set
            predicted_items: Predicted item set

        Returns:
            Item coverage reward
        """
        if not true_items:
            return 0.0

        intersection = predicted_items.intersection(true_items)
        hallucinated = predicted_items.difference(true_items)

        coverage = len(intersection) / len(true_items)

        if predicted_items:
            hallucination_penalty = self.lambda_hall * len(hallucinated) / len(predicted_items)
        else:
            hallucination_penalty = 0.0

        reward = coverage - hallucination_penalty
        return max(reward, 0.0)  # Clip to non-negative

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def compute_nutrition_reward(
        self,
        predicted_nutrition: Dict[str, Dict[str, float]],
        ground_truth_nutrition: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute nutritional accuracy reward using cosine similarity.

        Args:
            predicted_nutrition: Dict mapping items to nutrient vectors
            ground_truth_nutrition: Ground truth nutritional info

        Returns:
            Nutritional accuracy reward (cosine similarity)
        """
        if not ground_truth_nutrition or not predicted_nutrition:
            return 0.0

        similarities = []
        nutrient_keys = ['calories', 'protein', 'fat', 'carbs']

        for item, true_nutrients in ground_truth_nutrition.items():
            if item not in predicted_nutrition:
                continue

            pred_nutrients = predicted_nutrition[item]

            # Build vectors
            true_vec = np.array([true_nutrients.get(k, 0.0) for k in nutrient_keys])
            pred_vec = np.array([pred_nutrients.get(k, 0.0) for k in nutrient_keys])

            sim = self.cosine_similarity(true_vec, pred_vec)
            similarities.append(sim)

        if not similarities:
            return 0.0

        return float(np.mean(similarities))

    def compute_hallucination_penalty(
        self,
        predicted_banks: List[FoodBankEntry]
    ) -> float:
        """
        Compute hallucination penalty for non-existent food banks.

        Args:
            predicted_banks: List of predicted food banks

        Returns:
            Penalty value (0 if no hallucination, negative if hallucinated)
        """
        if not predicted_banks:
            return 0.0

        verified_count = sum(
            1 for bank in predicted_banks
            if bank.name in self.verified_registry
        )

        # Hallucination penalty in [-2, 0] (paper Appendix C)
        # r_hall = -2 * (1 - verification_rate); 0 = all verified, -2 = all hallucinated
        verification_rate = verified_count / len(predicted_banks)
        return -2.0 * (1.0 - verification_rate)

    def compute_reward(
        self,
        query_zipcode: str,
        predicted_banks: List[FoodBankEntry],
        ground_truth_banks: Optional[List[FoodBankEntry]] = None
    ) -> Tuple[float, RewardComponents]:
        """
        Compute total reward for a prediction.

        Args:
            query_zipcode: User's query ZIP code
            predicted_banks: Model's predicted food banks
            ground_truth_banks: Ground truth (for offline training)

        Returns:
            (total_reward, reward_components) tuple
        """
        # Compute individual components
        r_geo = self.compute_geo_reward(query_zipcode, predicted_banks)

        if ground_truth_banks:
            # Aggregate all items from ground truth
            true_items = set()
            true_nutrition = {}
            for bank in ground_truth_banks:
                true_items.update(bank.items)
                true_nutrition.update(bank.nutritional_info)

            # Aggregate predicted items
            pred_items = set()
            pred_nutrition = {}
            for bank in predicted_banks:
                pred_items.update(bank.items)
                pred_nutrition.update(bank.nutritional_info)

            r_items = self.compute_items_reward(true_items, pred_items)
            r_nutr = self.compute_nutrition_reward(pred_nutrition, true_nutrition)
        else:
            # Without ground truth, use heuristics
            r_items = 0.5  # Neutral
            r_nutr = 0.5

        r_hall = self.compute_hallucination_penalty(predicted_banks)

        components = RewardComponents(
            r_geo=r_geo,
            r_items=r_items,
            r_nutr=r_nutr,
            r_hall=r_hall
        )

        total_reward = components.aggregate(self.weights)

        return total_reward, components


def parse_output(output_text: str) -> List[FoodBankEntry]:
    """
    Parse model output into structured FoodBankEntry objects.

    Expected format:
    Food Bank Name1, zipcode1: Food Items1, Nutritional Information1; Food Items2, ...

    Args:
        output_text: Raw model output

    Returns:
        List of FoodBankEntry objects
    """
    entries = []

    # Split by food bank (semicolon followed by capitalized name pattern)
    bank_pattern = r'([^;]+?),\s*(\d{5}):\s*(.+?)(?=(?:[A-Z][^;]+?,\s*\d{5}:)|$)'
    matches = re.findall(bank_pattern, output_text, re.DOTALL)

    for bank_name, zipcode, items_text in matches:
        bank_name = bank_name.strip()
        items = []
        nutritional_info = {}

        # Parse items (split by semicolon)
        item_entries = items_text.split(';')
        for item_entry in item_entries:
            item_entry = item_entry.strip()
            if not item_entry:
                continue

            # Extract item name and nutrition
            # Format: "Item Name, calories: X, protein: Y, fat: Z, carbs: W"
            parts = item_entry.split(',')
            if parts:
                item_name = parts[0].strip()
                items.append(item_name)

                # Parse nutritional info
                nutrients = {}
                for part in parts[1:]:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip().lower()
                        try:
                            # Extract numeric value
                            num_match = re.search(r'([\d.]+)', value)
                            if num_match:
                                nutrients[key] = float(num_match.group(1))
                        except:
                            continue

                if nutrients:
                    nutritional_info[item_name] = nutrients

        entry = FoodBankEntry(
            name=bank_name,
            zipcode=zipcode,
            items=items,
            nutritional_info=nutritional_info
        )
        entries.append(entry)

    return entries
