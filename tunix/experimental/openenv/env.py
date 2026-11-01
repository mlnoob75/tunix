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

"""A Tunix environment that runs any text environment served by OpenEnv."""

import asyncio
from typing import Any, Dict, Optional

from tunix.experimental.openenv import _loop
from tunix.experimental.openenv import codecs as codecs_lib
from tunix.experimental.openenv import pool as pool_lib
from tunix.experimental.openenv import spec as spec_lib
from tunix.rl.agentic.environments import base_environment

# OpenEnv WebSocket error codes that mean "this action was bad", not "the
# session or server is broken".
_ACTION_ERROR_CODES = ("VALIDATION_ERROR", "EXECUTION_ERROR")


class OpenEnvTaskEnv(base_environment.BaseTaskEnv):
  """Runs one episode of an OpenEnv environment for Tunix's agentic learners.

  Pass the class as `env_class` and the server settings as `env_kwargs`:

    from tunix.experimental.openenv.env import OpenEnvTaskEnv

    learner = AgenticGRPOLearner(
        ...,
        agent_class=ModelAgent,
        env_class=OpenEnvTaskEnv,
        env_kwargs={"base_url": "http://localhost:8000", "max_steps": 10},
    )

  Observations are strings, so the stock `ModelAgent` works: the first
  observation holds the action instructions from the codec. Each step reward
  is the server's reward (`None` counts as 0). Tunix sums them per trajectory.

  Each dataset example becomes the `reset()` arguments: the example's
  `reset_kwargs` dict if it has one, else the example keys that the server's
  reset schema accepts (for example `seed`). Use the same arguments for every
  member of a GRPO group so that they all get the same task.

  Args:
    task: The dataset example. Tunix passes it positionally.
    base_url: URL of the OpenEnv server (local, Docker or a Hugging Face Space).
    codec: A `Codec`. Default: the codec registered for the server's
      environment name, else `ToolCallCodec` (MCP) or `SchemaCodec` (typed).
    max_steps: Episode length limit, counted in agent turns.
    reset_kwargs: `reset()` arguments shared by all episodes. Task-derived
      arguments override them.
    max_sessions: Concurrent sessions to this server. Match the server's
      `MAX_CONCURRENT_ENVS`.
    invalid_action_reward: Reward for a reply that is not a valid action.
    request_timeout: Seconds for one reset or step round trip.
    acquire_timeout: Seconds to wait for a free session.
    headers: HTTP headers, for example a token for a private Space.
    **kwargs: Passed to `BaseTaskEnv` (Tunix adds `group_id`, `pair_index`).
  """

  def __init__(
      self,
      task: Optional[Dict[str, Any]] = None,
      *,
      base_url: str,
      codec: Optional[codecs_lib.Codec] = None,
      max_steps: int = 8,
      reset_kwargs: Optional[Dict[str, Any]] = None,
      max_sessions: int = 8,
      invalid_action_reward: float = 0.0,
      request_timeout: float = 120.0,
      acquire_timeout: float = 1800.0,
      headers: Optional[Dict[str, str]] = None,
      **kwargs,
  ):
    super().__init__(task=task, max_steps=max_steps, **kwargs)
    self.spec = spec_lib.discover_cached(base_url, headers)
    self.codec = codec or codecs_lib.codec_for(self.spec)
    self._reset_kwargs = dict(reset_kwargs or {})
    self._invalid_action_reward = invalid_action_reward
    self._request_timeout = request_timeout
    self._acquire_timeout = acquire_timeout
    self._pool = pool_lib.get_pool(
        self.spec.base_url, max_sessions, headers, request_timeout
    )
    self._client = None
    self._healthy = True

  # ------------------------------------------------------------------ reset

  def episode_reset_kwargs(self) -> Dict[str, Any]:
    """The `reset()` arguments for this episode."""
    kwargs = dict(self._reset_kwargs)
    if isinstance(self.task.get("reset_kwargs"), dict):
      kwargs.update(self.task["reset_kwargs"])
    else:
      accepted = set(self.spec.reset_params)
      kwargs.update({k: v for k, v in self.task.items() if k in accepted})
    return kwargs

  def _initial_observation(self) -> str:
    self._release()
    self._client = _loop.run(self._pool.acquire(), self._acquire_timeout)
    result = self._call("reset", **self.episode_reset_kwargs())
    self._last_observation = result.observation
    return (
        self.codec.instructions(self.spec)
        + "\n\n"
        + self.codec.render_observation(result.observation, self.spec)
    )

  # ------------------------------------------------------------------- step

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    if self._client is None:
      raise RuntimeError("step() called before reset() or after close().")
    # Tunix agents return `agent_types.Action`; the text is in `.action`.
    text = getattr(action, "action", action)
    text = text if isinstance(text, str) else str(text)
    try:
      payload = self.codec.parse_action(text, self.spec)
    except codecs_lib.ActionParseError as e:
      return base_environment.EnvStepResult(
          observation=f"Invalid action: {e}",
          reward=self._invalid_action_reward,
          done=False,
          info={"openenv": {"invalid_action": str(e)}},
      )
    try:
      result = self._call("step", payload)
    except RuntimeError as e:
      # The server rejected the action: it failed the action schema, or the
      # environment raised while running it. The session is still usable, so
      # tell the model and go on. Other server errors end the trajectory.
      if not any(f"(code: {code})" in str(e) for code in _ACTION_ERROR_CODES):
        self._healthy = False
        raise
      return base_environment.EnvStepResult(
          observation=(
              f"Invalid action: the environment rejected {payload}. ({e})"
          ),
          reward=self._invalid_action_reward,
          done=False,
          info={"openenv": {"rejected_action": payload, "error": str(e)}},
      )
    self._last_observation = result.observation
    if result.done:
      self._release()
    return base_environment.EnvStepResult(
        observation=self.codec.render_observation(
            result.observation, self.spec
        ),
        reward=float(result.reward or 0.0),
        done=bool(result.done),
        info={
            "openenv": {
                "action": payload,
                "observation": result.observation,
                "metadata": result.metadata,
            }
        },
    )

  # -------------------------------------------------------------- lifecycle

  def _call(self, method: str, *args, **kwargs):
    """Calls `self._client.<method>` on the shared loop.

    The call itself must happen on the loop: OpenEnv clients run synchronously
    when no event loop is running in the calling thread.
    """
    client = self._client

    async def call():
      return await getattr(client, method)(*args, **kwargs)

    try:
      return _loop.run(call(), self._request_timeout)
    except (ConnectionError, asyncio.TimeoutError, TimeoutError):
      self._healthy = False
      raise

  def _release(self):
    if self._client is not None:
      client, self._client = self._client, None
      _loop.run(self._pool.release(client, self._healthy))
      self._healthy = True

  def close(self):
    self._release()
