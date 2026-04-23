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
import sys
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

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> Food4AllRolePlaying:
    r"""Construct a Food4All Planner-Executor society for free food discovery.

    Implements the dual-agent architecture from the Food4All paper (arXiv:2510.18289):
    - Planner Agent (A_P): hierarchical task decomposition
    - Executor Agent (A_E): tool-based retrieval across heterogeneous data sources

    Args:
        question (str): User query, typically "I live in [ZIP], where can I get
            free food nearby? What nutrients are provided?"

    Returns:
        Food4AllRolePlaying: Configured dual-agent society.
    """
    # Create models for different components
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

    # Executor tools — focus on food bank discovery (paper Figure 2)
    tools = [
        *BrowserToolkit(
            headless=True,  # headless for server deployment
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_wiki,
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    planner_agent_kwargs = {"model": models["planner"]}
    executor_agent_kwargs = {"model": models["executor"], "tools": tools}

    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Food4AllRolePlaying uses food-specific Planner/Executor system prompts
    society = Food4AllRolePlaying(
        **task_kwargs,
        user_role_name="Planner",
        user_agent_kwargs=planner_agent_kwargs,
        assistant_role_name="Executor",
        assistant_agent_kwargs=executor_agent_kwargs,
    )

    return society


def main():
    r"""Main function to run Food4All with a default food access query."""
    # Default food access query (paper example, Section 5.5)
    default_task = (
        "I live in 94102, where can I get free food nearby? "
        "What nutrients are provided? Please list specific food items with "
        "their nutritional information (calories, protein, fat, carbohydrates)."
    )

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Construct and run the society
    society = construct_society(task)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
