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

"""Food4All: multi-agent framework for real-time free food discovery.

Implements the full pipeline from arXiv:2510.18289:
  - Dual-agent (Planner + Executor) with T_max=15 rounds, J_max=25K token budget
  - Multi-component reward function (r_geo, r_items, r_nutr, r_hall)
  - Offline RL via DPO (β=0.2, weights=(0.3,0.3,0.3,0.1), λ=0.4)
  - Online RL via pairwise preference or questionnaire feedback
"""

import sys
import re
import pathlib
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    SearchToolkit,
    BrowserToolkit,
    FileWriteToolkit,
    DocumentProcessingToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level

from owl.utils import run_society, Food4AllRolePlaying
from owl.utils.reward_function import RewardFunction, parse_output
from owl.utils.online_rl import OnlineRLLoop, OnlineFeedbackBuffer

# ---------------------------------------------------------------------------
# Config constants from Table 2 / paper Section 3
# ---------------------------------------------------------------------------
T_MAX = 15          # Maximum dialogue rounds (paper default)
J_MAX = 25_000      # Token budget (paper: J_max = 25K)

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> Food4AllRolePlaying:
    """Construct the Food4All Planner-Executor society.

    Args:
        question: User query, e.g. "I live in 94102, where can I get free food
            nearby? What nutrients are provided?"

    Returns:
        Food4AllRolePlaying society ready for execution.
    """
    models = {
        "planner": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "executor": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
    }

    # Executor tools — heterogeneous data aggregation (paper Section 3.1, Figure 2)
    tools = [
        *BrowserToolkit(
            headless=True,
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_wiki,
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    society = Food4AllRolePlaying(
        task_prompt=question,
        with_task_specify=False,
        user_role_name="Planner",
        user_agent_kwargs={"model": models["planner"]},
        assistant_role_name="Executor",
        assistant_agent_kwargs={"model": models["executor"], "tools": tools},
    )
    return society


def check_token_budget(token_info: dict) -> bool:
    """Return True if within the J_max=25K token budget."""
    total = (
        token_info.get("completion_token_count", 0)
        + token_info.get("prompt_token_count", 0)
    )
    return total <= J_MAX


def run_food4all(
    question: str,
    enable_online_rl: bool = False,
    feedback_mode: str = "questionnaire",
) -> tuple[str, list, dict]:
    """Run the full Food4All pipeline for a single query.

    Args:
        question: Free-food access query from the user.
        enable_online_rl: Whether to collect user feedback for online RL update.
        feedback_mode: "pairwise" or "questionnaire" (paper Section 3.3).

    Returns:
        (answer, chat_history, token_info) tuple.
    """
    society = construct_society(question)
    answer, chat_history, token_info = run_society(society, round_limit=T_MAX)

    if not check_token_budget(token_info):
        print(
            f"[Food4All] Token budget J_max={J_MAX} exceeded: "
            f"{token_info.get('completion_token_count',0) + token_info.get('prompt_token_count',0)} tokens used."
        )

    if enable_online_rl:
        _run_online_rl_feedback(question, answer, feedback_mode)

    return answer, chat_history, token_info


def _run_online_rl_feedback(
    question: str,
    answer: str,
    feedback_mode: str,
) -> None:
    """Collect user feedback and update the online RL policy (paper Section 3.3)."""
    reward_fn = RewardFunction()
    buffer = OnlineFeedbackBuffer(max_size=5000)
    rl_loop = OnlineRLLoop(
        reward_function=reward_fn,
        feedback_buffer=buffer,
        beta=0.1,
        learning_rate=5e-6,
    )

    zipcode = _extract_zipcode(question)

    if feedback_mode == "pairwise":
        # Generate a second candidate via a fresh society run
        society2 = construct_society(question)
        answer2, _, _ = run_society(society2, round_limit=T_MAX)

        feedback = rl_loop.collect_pairwise_feedback(question, answer, answer2)
        metrics = rl_loop.update_policy(feedback, zipcode)
        print(f"[OnlineRL] Pairwise update metrics: {metrics}")
    else:
        feedback = rl_loop.collect_questionnaire_feedback(question, answer)
        metrics = rl_loop.update_policy(feedback, zipcode)
        print(f"[OnlineRL] Questionnaire update metrics: {metrics}")


def _extract_zipcode(query: str) -> str:
    """Extract the first 5-digit ZIP code from the query."""
    match = re.search(r"\b(\d{5})\b", query)
    return match.group(1) if match else "00000"


def main():
    """Run Food4All with an example food access query."""
    # Paper case study example (Section 5.5, Table 3)
    default_task = (
        "I live in 94102, where can I get free food nearby? "
        "What nutrients are provided? Please list specific food items with "
        "their nutritional information (calories, protein, fat, carbohydrates)."
    )

    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Set enable_online_rl=True to activate the online RL feedback loop
    answer, chat_history, token_info = run_food4all(
        task,
        enable_online_rl=False,
        feedback_mode="questionnaire",
    )

    completion = token_info.get("completion_token_count", 0)
    prompt = token_info.get("prompt_token_count", 0)
    print(f"\033[94mAnswer:\n{answer}\033[0m")
    print(
        f"\033[93mTokens — completion: {completion:,} | prompt: {prompt:,} | "
        f"total: {completion + prompt:,} / {J_MAX:,}\033[0m"
    )


if __name__ == "__main__":
    main()
