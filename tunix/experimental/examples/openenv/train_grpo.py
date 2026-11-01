# Copyright 2026 The Tunix Authors.
#
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

"""Agentic GRPO on an OpenEnv environment, for Colab or a TPU VM.

NOT RUN. This was written against Tunix at commit c4c5da1 (2026-10-30) but
never executed, because training needs a TPU. The adapter itself is tested on
CPU against local OpenEnv servers (tests/experimental/openenv/).

How to use it:

1. Install the extra and start the OpenEnv server next to the trainer (or
   point at a HF Space):

     pip install "google-tunix[openenv]"

     pip install -e path/to/OpenEnv/envs/reasoning_gym_env
     MAX_CONCURRENT_ENVS=32 python -m uvicorn \
         reasoning_gym_env.server.app:app --port 8000 &

2. Copy Tunix's examples/frozenlake/train_frozenlake.py. It builds the model,
   meshes, optimizer, cluster config and GRPOConfig. Keep all of that.

3. Replace its `create_datasets()` with `create_datasets()` below, and its
   `GRPOLearner(...)` call with `make_learner(...)` below. Remove the
   FrozenLakeAgent and FrozenLakeEnv imports.

Pick `MAX_SESSIONS` equal to the server's MAX_CONCURRENT_ENVS. It is fine for
GRPOConfig.max_concurrency to be larger: extra episodes wait for a session.
"""

import grain
from tunix.experimental.openenv import codecs
from tunix.experimental.openenv import env as openenv_env
from tunix.experimental.openenv import spec
from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner
from tunix.rl.agentic.agents.model_agent import ModelAgent

BASE_URL = "http://localhost:8000"
MAX_SESSIONS = 32

# reasoning_gym builds its task set from these reset() arguments. The
# per-example `seed` picks the task, so all members of a GRPO group get the
# same one.
RESET_KWARGS = {
    "dataset_name": "leg_counting",
    "dataset_config": {},
    "size": 10_000,
}


def create_datasets(num_train: int = 4096, num_test: int = 256):
  """Each example is only a reset seed. `prompts` is required by Tunix and
  stays empty: the first observation from the env is the prompt."""

  def rows(start, count):
    return [{"prompts": "", "seed": start + i} for i in range(count)]

  train = grain.MapDataset.source(rows(0, num_train))
  test = grain.MapDataset.source(rows(1_000_000, num_test))
  return train, test


def make_learner(rl_engine, grpo_config, chat_parser, metric_fns=()):
  # Fail early, before the model is on the TPU, if the server is not up.
  env_spec = spec.discover(BASE_URL)
  print(
      f"OpenEnv: {env_spec.name} ({env_spec.kind}),"
      f" reset params {env_spec.reset_params}"
  )
  return GRPOLearner(
      rl_engine=rl_engine,
      agent_class=ModelAgent,
      agent_kwargs={},
      env_class=openenv_env.OpenEnvTaskEnv,
      env_kwargs={
          "base_url": BASE_URL,
          "max_steps": (
              1
          ),  # reasoning_gym is single-turn. Use ~8-40 for maze etc.
          "max_sessions": MAX_SESSIONS,
          "reset_kwargs": RESET_KWARGS,
          # The answer and score are returned after the step; the model never
          # sees them because the episode ends there. Hide them anyway.
          "codec": codecs.SchemaCodec(
              hide_fields=["correct_answer", "dataset_metadata"]
          ),
      },
      algo_config=grpo_config,
      chat_parser=chat_parser,
      metric_fns=list(metric_fns),
  )


# In the copied script:
#
#   train_dataset, test_dataset = create_datasets()
#   ... (post_init_dataset, model, rl_engine, grpo_config as in the example)
#   from tunix.experimental.openenv import pool
#
#   grpo_trainer = make_learner(
#       rl_engine, grpo_config, chat_parser, [metric_fn]
#   )
#   try:
#     grpo_trainer.train(train_dataset, eval_dataset=test_dataset)
#   finally:
#     pool.close_all()
#
# Note: the copied example sets GRPOConfig.system_prompt="". The codec puts the
# action instructions in the first user message, so that still works.
