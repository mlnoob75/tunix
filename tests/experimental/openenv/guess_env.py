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

"""A small MCP environment with rewards, for offline tests.

The server picks a number from 1 to 20 (from the reset seed). The `guess` tool
answers "higher", "lower" or "correct". A correct guess gives reward 1 and ends
the episode, the same way finqa_env rewards its `submit_answer` tool.

Run from this directory: python -m uvicorn guess_env:app --port 8000
"""

import random
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.mcp_types import CallToolObservation
from openenv.core.env_server.types import Observation
from openenv.core.env_server.types import State


class GuessEnvironment(MCPEnvironment):
  """Hidden number from the reset seed; reward 1 for the right guess."""

  SUPPORTS_CONCURRENT_SESSIONS = True

  def __init__(self):
    mcp = FastMCP("guess_env")

    @mcp.tool
    def guess(number: int) -> str:
      """Guess the hidden number (1 to 20).

      Args:
        number: Your guess.
      """
      if number < self._target:
        return "higher"
      if number > self._target:
        return "lower"
      return "correct"

    super().__init__(mcp)
    self._target = 0
    self._state = State(episode_id=str(uuid4()), step_count=0)

  def reset(
      self,
      seed: Optional[int] = None,
      episode_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Observation:
    self._target = random.Random(seed).randint(1, 20)
    self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
    return Observation(
        done=False,
        reward=0.0,
        metadata={"task": "Find the number from 1 to 20."},
    )

  def _score(self, action, obs):
    if isinstance(action, CallToolAction) and action.tool_name == "guess":
      if action.arguments.get("number") == self._target:
        obs.reward, obs.done = 1.0, True
    return obs

  def step(self, action, timeout_s=None, **kwargs):
    self._state.step_count += 1
    return self._score(
        action, super().step(action, timeout_s=timeout_s, **kwargs)
    )

  async def step_async(self, action, timeout_s=None, **kwargs):
    self._state.step_count += 1
    obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
    return self._score(action, obs)

  def _step_impl(self, action, timeout_s=None, **kwargs):
    raise ValueError(f"Unsupported action: {type(action).__name__}")

  @property
  def state(self) -> State:
    return self._state


app = create_app(
    GuessEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="guess_env",
    max_concurrent_envs=4,
)
