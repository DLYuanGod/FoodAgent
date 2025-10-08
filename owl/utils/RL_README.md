# Food4All Reinforcement Learning Framework

This directory contains the RL components for the Food4All system as described in the paper.

## Components

### 1. **Reward Function** (`reward_function.py`)
Multi-component reward function with:
- **Geographic reward** (r_geo): Measures proximity to target ZIP code
- **Item coverage reward** (r_items): Evaluates food item accuracy with hallucination penalty
- **Nutritional accuracy reward** (r_nutr): Cosine similarity of nutrient vectors
- **Hallucination penalty** (r_hall): Penalizes non-existent food banks

### 2. **Offline RL** (`offline_rl.py`)
Direct Preference Optimization (DPO) for bootstrapping before deployment:
- Preference dataset loading from curated cases
- DPO loss computation
- Negative sample generation via corruption
- Training and evaluation loops

### 3. **Online RL** (`online_rl.py`)
Continual adaptation with user feedback:
- **Pairwise preference collection**: User chooses between two outputs
- **Questionnaire-based feedback**: Structured questions about accuracy
- Adaptive weight tuning via EMA
- Policy gradient updates with variance reduction

## Usage

### Basic Inference (without RL)
```bash
cd examples
python run_deepseek_zh.py
```

### With Online RL (Questionnaire Mode)
```bash
export ENABLE_ONLINE_RL=true
export FEEDBACK_MODE=questionnaire
python run_deepseek_zh.py
```

### With Online RL (Pairwise Preference Mode)
```bash
export ENABLE_ONLINE_RL=true
export FEEDBACK_MODE=pairwise
python run_deepseek_zh.py
```

### Offline Training Example
```python
from owl.utils.reward_function import RewardFunction
from owl.utils.offline_rl import OfflineRLTrainer

# Initialize reward function
reward_fn = RewardFunction()

# Initialize trainer
trainer = OfflineRLTrainer(
    reward_function=reward_fn,
    beta=0.1,
    learning_rate=1e-5,
)

# Load dataset
dataset = trainer.load_dataset('path/to/preference_data.json')

# Train
metrics = trainer.train_epoch(dataset, verbose=True)
print(f"Training metrics: {metrics}")
```

### Dataset Format (Offline RL)
```json
[
  {
    "query": "I live in the 46637 area. Please help me find food banks...",
    "positive": "Food Bank A, 46637: bread (calories: 250/100g, protein: 8g/100g, ...)",
    "negative": "Food Bank X, 99999: fake_item (calories: 0, ...)",
    "ground_truth": [
      {
        "name": "Food Bank A",
        "zipcode": "46637",
        "items": ["bread", "milk"],
        "nutritional_info": {
          "bread": {"calories": 250, "protein": 8, "fat": 1, "carbs": 50}
        },
        "latitude": 41.7,
        "longitude": -86.2
      }
    ]
  }
]
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│          User Query (ZIP code)                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│       Planner Agent (Task Decomposition)        │
│  • Search food banks                            │
│  • Mine Reddit for food lists                   │
│  • Fetch nutritional info                       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│      Executor Agent (Tool Invocation)           │
│  • Wikipedia search                             │
│  • Reddit scraping                              │
│  • USDA API calls                               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Structured Output                       │
│  Bank1, ZIP1: Item1 (nutrition); Item2...       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Reward Computation                      │
│  R = w1·r_geo + w2·r_items + w3·r_nutr + w4·r_hall
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│  Offline RL  │  │   Online RL      │
│    (DPO)     │  │ (User Feedback)  │
└──────────────┘  └──────────────────┘
```

## Computational Efficiency

- **Deterministic decoding** (temperature=0): Enables KV-cache reuse
- **Lazy tool initialization**: Only load tools when needed
- **Token budget tracking**: Median 17.3K tokens per query
- **Termination control**: Max 15 rounds or 25K token budget

## Reward Function Details

### Geographic Reward
```python
r_geo = exp(-avg_distance / 50km)
```

### Item Coverage Reward
```python
r_items = |I_pred ∩ I_true| / |I_true| - λ * |I_pred \ I_true| / |I_pred|
```

### Nutritional Accuracy Reward
```python
r_nutr = cosine_similarity(ν_pred, ν_true)
```

### Hallucination Penalty
```python
r_hall = |verified_banks| / |total_predicted_banks|
```

## Feedback Collection

### Questionnaire Format
1. Was the information accurate? (yes/no)
2. Which parts were incorrect?
   - Location/ZIP code
   - Food items
   - Nutritional information
   - Food bank names (hallucination)
3. Additional comments (optional)

### Pairwise Preference
- User is shown two candidate outputs
- Selects preferred output
- System learns from preference signal

## References

See the Food4All paper for full algorithm details and theoretical foundations.
