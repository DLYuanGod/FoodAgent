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


# To run this file, you need to configure the OpenAI-compatible API for GPT-OSS 20B
# Set OPENAI_API_BASE_URL and OPENAI_API_KEY in your .env file

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

    # Create models for GPT-OSS 20B (using OpenAI-compatible endpoint)
    # Note: GPT-OSS 20B would be hosted via an OpenAI-compatible API
    # You need to set OPENAI_API_BASE_URL to point to your GPT-OSS 20B server
    # and OPENAI_API_KEY for authentication

    models = {
        "planner": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="gpt-oss-20b",  # This should match your model name
            model_config_dict={
                "temperature": 0,  # Deterministic decoding for KV-cache reuse
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_API_BASE_URL"),
            },
        ),
        "executor": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="gpt-oss-20b",
            model_config_dict={
                "temperature": 0,  # Deterministic decoding
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_API_BASE_URL"),
            },
        ),
    }

    # Configure toolkits for Executor Agent
    # As per the paper, only the Executor has access to tools
    tools = [
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        SearchToolkit().search_wiki,  # Wikipedia search for food banks
        # SearchToolkit().search_google,  # Optional: Google search
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),  # For audit artifacts
        *RedditToolkit().get_tools(),  # Social mining for food lists
    ]

    # Configure agent roles and parameters
    # Planner Agent: task decomposition (no tools)
    planner_agent_kwargs = {
        "model": models["planner"],
    }

    # Executor Agent: tool invocation (with tools)
    executor_agent_kwargs = {
        "model": models["executor"],
        "tools": tools
    }

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    # Planner = user role, Executor = assistant role
    society = RolePlaying(
        **task_kwargs,
        user_role_name="Planner Agent",
        user_agent_kwargs=planner_agent_kwargs,
        assistant_role_name="Executor Agent",
        assistant_agent_kwargs=executor_agent_kwargs
    )

    return society


def main():
    r"""Main function to run the Food4All system with GPT-OSS 20B."""

    # Default Food4All task
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

    # Configuration options from environment
    enable_online_rl = os.environ.get('ENABLE_ONLINE_RL', 'false').lower() == 'true'
    feedback_mode = os.environ.get('FEEDBACK_MODE', 'questionnaire')  # 'pairwise' or 'questionnaire'
    max_rounds = int(os.environ.get('MAX_ROUNDS', '15'))  # T_max from paper
    max_tokens = int(os.environ.get('MAX_TOKENS', '25000'))  # J_max from paper

    print("="*80)
    print("Food4All System - GPT-OSS 20B")
    print("="*80)
    print(f"Model: GPT-OSS 20B")
    print(f"Max Rounds: {max_rounds}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Online RL: {enable_online_rl}")
    if enable_online_rl:
        print(f"Feedback Mode: {feedback_mode}")
    print("="*80)

    # Construct and run the society
    society = construct_society(task)

    answer, chat_history, token_count = run_society(society)

    # Output the result
    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)
    print(f"\033[94m{answer}\033[0m")
    print("="*80)
    print(f"Total Tokens Used: {token_count}")
    print("="*80)

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
            feedback_path = feedback_dir / "online_feedback_gpt_oss.json"
            feedback_buffer.save(str(feedback_path))
            print(f"\n✓ Feedback saved to: {feedback_path}")

        except KeyboardInterrupt:
            print("\n\nOnline RL interrupted by user.")
        except Exception as e:
            print(f"\n\nError in online RL: {e}")

    # Parse and analyze output
    print("\n" + "="*80)
    print("OUTPUT ANALYSIS")
    print("="*80)

    parsed_banks = parse_output(answer)
    if parsed_banks:
        print(f"\033[92m✓ Successfully parsed {len(parsed_banks)} food bank(s)\033[0m\n")
        for i, bank in enumerate(parsed_banks, 1):
            print(f"{i}. {bank.name}")
            print(f"   ZIP Code: {bank.zipcode}")
            print(f"   Items: {len(bank.items)}")
            if bank.items:
                print(f"   Sample Items: {', '.join(bank.items[:3])}")
                if len(bank.items) > 3:
                    print(f"                 ... and {len(bank.items) - 3} more")
            print()
    else:
        print("\033[91m✗ Could not parse any food banks from output\033[0m")
        print("Please check output format.")

    # Save audit artifacts (as mentioned in paper)
    artifacts_dir = pathlib.Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save chat history
    import json
    chat_history_path = artifacts_dir / "chat_history.json"
    with open(chat_history_path, 'w') as f:
        json.dump(chat_history, f, indent=2, default=str)
    print(f"✓ Chat history saved to: {chat_history_path}")

    # Save final answer
    answer_path = artifacts_dir / "final_answer.txt"
    with open(answer_path, 'w') as f:
        f.write(answer)
    print(f"✓ Final answer saved to: {answer_path}")

    print("="*80)


if __name__ == "__main__":
    main()
