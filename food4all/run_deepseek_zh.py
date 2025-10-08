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


# To run this file, you need to configure the DeepSeek API key
# You can obtain your API key from DeepSeek platform: https://platform.deepseek.com/api_keys
# Set it as DEEPSEEK_API_KEY="your-api-key" in your .env file or add it to your environment variables

import sys
from dotenv import load_dotenv

from camel.models import ModelFactory
from camel.toolkits import (
    ExcelToolkit,
    SearchToolkit,
    FileWriteToolkit,
    CodeExecutionToolkit,
)
from camel.toolkits import RedditToolkit
from camel.types import ModelPlatformType, ModelType
from camel.societies import RolePlaying
from camel.logger import set_log_level

from owl.utils import run_society
from owl.utils.reward_function import RewardFunction, parse_output
from owl.utils.online_rl import OnlineRLLoop, OnlineFeedbackBuffer

import pathlib
import os

set_log_level(level="DEBUG")

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))


def construct_society(question: str) -> RolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
    }

    # Configure toolkits
    tools = [
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        # SearchToolkit().search_google,
        SearchToolkit().search_wiki,
        # SearchToolkit().search_baidu,
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
        *RedditToolkit().get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = RolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs
    )

    return society


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    default_task = """
I live in the 46637 area. Please help me complete the following three-step task:

1. Search for local food banks:
   Go to Wiki and search for free food banks or food pantries near the area code.
   List at least 3 options with their names, addresses, and websites (if available).

2. Find recent food lists:
   Go to Reddit and look for recent posts or discussions that mention what items are typically available at these food banks. Your search should be align with the food banks you found in the previous step. Please try your best to find the full food list.

3. Analyze nutritional value:
   For each food item found in those lists, provide the basic nutritional information—such as calories, protein, fat,
   carbohydrates, and key vitamins/minerals (you may use public nutritional databases like USDA for reference). Please specify the unit, for example 100 calories/g.

Final Step: Please output this format: Food Bank Name1, zipcode1: Food Items1, Nutritional Information1; Food Items2, Nutritional Information2; .... ; Food Bank Name2, zipcode2: Food Items1, Nutritional Information1; Food Items2, Nutritional Information2; .... ; ....
"""

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Check if online RL mode is enabled
    enable_online_rl = os.environ.get('ENABLE_ONLINE_RL', 'false').lower() == 'true'
    feedback_mode = os.environ.get('FEEDBACK_MODE', 'questionnaire')  # 'pairwise' or 'questionnaire'

    # Construct and run the society
    society = construct_society(task)

    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")

    # Online RL loop (optional)
    if enable_online_rl:
        print("\n" + "="*80)
        print("ONLINE RL MODE ENABLED")
        print("="*80)

        # Initialize reward function and online RL loop
        reward_function = RewardFunction()
        feedback_buffer = OnlineFeedbackBuffer()
        online_rl = OnlineRLLoop(
            reward_function=reward_function,
            feedback_buffer=feedback_buffer,
            beta=0.1,
        )

        # Define function to generate outputs
        def generate_output(query: str) -> str:
            society = construct_society(query)
            ans, _, _ = run_society(society)
            return ans

        # Run online RL loop
        try:
            metrics = online_rl.run_online_loop(
                query=task,
                generate_output_fn=generate_output,
                num_iterations=1,
                feedback_mode=feedback_mode,
                verbose=True,
            )

            # Save feedback buffer
            feedback_dir = pathlib.Path(__file__).parent.parent / "feedback_data"
            feedback_dir.mkdir(exist_ok=True)
            feedback_path = feedback_dir / "online_feedback.json"
            feedback_buffer.save(str(feedback_path))
            print(f"\n✓ Feedback saved to: {feedback_path}")

        except KeyboardInterrupt:
            print("\n\nOnline RL interrupted by user.")
        except Exception as e:
            print(f"\n\nError in online RL: {e}")

    # Parse and analyze output
    parsed_banks = parse_output(answer)
    if parsed_banks:
        print(f"\n\033[92m✓ Parsed {len(parsed_banks)} food bank(s) from output\033[0m")
        for i, bank in enumerate(parsed_banks, 1):
            print(f"  {i}. {bank.name} ({bank.zipcode}) - {len(bank.items)} items")


if __name__ == "__main__":
    main()
