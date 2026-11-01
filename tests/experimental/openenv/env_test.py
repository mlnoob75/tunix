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

"""End-to-end tests of OpenEnvTaskEnv against local OpenEnv servers.

The servers run on free local ports: OpenEnv's echo_env (MCP),
reasoning_gym_env (typed, one turn), maze_env (typed, multi-turn), and
guess_env.py next to this file (MCP with rewards and a 4-session limit). Tests
for an environment package that is not installed are skipped.

`_run_episode` follows the call order of TrajectoryCollectEngine: `reset()` and
`step()` run in executor threads under `wait_for`, observations go through the
real `ModelAgent`, and a scripted policy stands in for the model.
"""

# The tests check the pool and session state directly.
# pylint: disable=protected-access

import asyncio
import collections
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request

from absl.testing import absltest

if "pytest" in sys.modules:
  try:
    import pytest  # pylint: disable=g-import-not-at-top

    pytest.importorskip("openenv")
  except ImportError:
    pass

try:
  import openenv  # pylint: disable=unused-import,g-import-not-at-top
except ImportError:
  raise unittest.SkipTest("openenv is not installed")  # pylint: disable=raise-missing-from

# pylint: disable=g-import-not-at-top,wrong-import-position
from tunix.experimental.openenv import codecs
from tunix.experimental.openenv import env as env_lib
from tunix.experimental.openenv import maze_codec
from tunix.experimental.openenv import pool as pool_lib
from tunix.rl.agentic.agents import model_agent

_HERE = os.path.dirname(os.path.abspath(__file__))

# Server name -> (uvicorn app, package that must be installed).
_SERVERS = {
    "echo": ("echo_env.server.app:app", "echo_env"),
    "guess": ("guess_env:app", "fastmcp"),
    "reasoning_gym": ("reasoning_gym_env.server.app:app", "reasoning_gym_env"),
    "maze": ("maze_env.server.app:app", "maze_env"),
}
_urls = {}
_processes = []


def _free_port():
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _serve(app):
  """Starts `app` with uvicorn on a free port and waits for /health."""
  port = _free_port()
  process = subprocess.Popen(
      [
          sys.executable,
          "-m",
          "uvicorn",
          app,
          "--port",
          str(port),
          "--log-level",
          "warning",
      ],
      cwd=_HERE,
      env={**os.environ, "MAX_CONCURRENT_ENVS": "8"},
  )
  url = f"http://127.0.0.1:{port}"
  deadline = time.time() + 60
  while time.time() < deadline:
    try:
      urllib.request.urlopen(f"{url}/health", timeout=1)
      return process, url
    except OSError as e:
      if process.poll() is not None:
        raise RuntimeError(
            f"{app} exited with code {process.returncode}"
        ) from e
      time.sleep(0.2)
  process.kill()
  raise RuntimeError(f"{app} did not start")


def setUpModule():
  """Starts a server for each environment package that is installed."""
  for name, (app, package) in _SERVERS.items():
    if importlib.util.find_spec(package) is not None:
      process, _urls[name] = _serve(app)
      _processes.append(process)


def tearDownModule():
  pool_lib.close_all()
  for process in _processes:
    process.terminate()
    process.wait(10)


def _url(name):
  if name not in _urls:
    raise unittest.SkipTest(f"{_SERVERS[name][1]} is not installed")
  return _urls[name]


def _call(name, arguments):
  body = json.dumps({"name": name, "arguments": arguments})
  return f"<tool_call>{body}</tool_call>"


def _last_user(messages):
  return [m for m in messages if m["role"] == "user"][-1]["content"]


def _users(agent):
  return [m["content"] for m in agent.chat_completions if m["role"] == "user"]


async def _run_episode(env, policy, timeout=60.0):
  """Runs one episode the way TrajectoryCollectEngine does."""
  loop = asyncio.get_running_loop()
  agent = model_agent.ModelAgent(system_prompt="You are a helpful agent.")
  agent.reset()
  obs, info = await asyncio.wait_for(
      loop.run_in_executor(None, env.reset), timeout
  )
  agent.update_from_env(obs, 0.0, False, info)
  rewards, done = [], False
  try:
    while not done:
      action = agent.update_from_model(policy(agent.chat_completions))
      obs, reward, done, info = await asyncio.wait_for(
          loop.run_in_executor(None, env.step, action), timeout
      )
      agent.update_from_env(obs, reward, done, info)
      rewards.append(reward)
  finally:
    await loop.run_in_executor(None, env.close)
  return agent, rewards


def _run(env, policy):
  """Runs one episode from synchronous test code."""
  return asyncio.run(_run_episode(env, policy))


def _binary_search_policy():
  """Returns a policy that solves guess_env by binary search."""
  low, high, guess = 1, 20, None

  def policy(messages):
    nonlocal low, high, guess
    feedback = _last_user(messages)
    if feedback == "higher":
      low = guess + 1
    elif feedback == "lower":
      high = guess - 1
    guess = (low + high) // 2
    return _call("guess", {"number": guess})

  return policy


def _bfs_policy(messages):
  """Reads the MazeCodec grid and moves one step along a shortest path."""
  grid = [
      line
      for line in _last_user(messages).splitlines()
      if set(line) <= set("#.AE")
  ]
  cells = {
      (r, c): ch for r, line in enumerate(grid) for c, ch in enumerate(line)
  }
  start = next(p for p, ch in cells.items() if ch == "A")
  moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
  queue, first_move = collections.deque([start]), {start: None}
  while queue:
    pos = queue.popleft()
    if cells[pos] == "E":
      return f"<move>{first_move[pos]}</move>"
    for name, (dr, dc) in moves.items():
      nxt = (pos[0] + dr, pos[1] + dc)
      if cells.get(nxt, "#") != "#" and nxt not in first_move:
        first_move[nxt] = first_move[pos] or name
        queue.append(nxt)
  raise AssertionError("no path to the exit")


class EchoTest(absltest.TestCase):

  def test_discovers_tools_and_calls_them(self):
    env = env_lib.OpenEnvTaskEnv(base_url=_url("echo"), max_steps=2)
    self.assertEqual(env.spec.kind, "mcp")
    self.assertEqual(
        {t["name"] for t in env.spec.tools},
        {"echo_message", "echo_with_length"},
    )
    replies = iter([
        _call("echo_with_length", {"message": "hello"}),
        _call("echo_message", {"message": "bye"}),
    ])
    agent, rewards = _run(env, lambda messages: next(replies))
    users = _users(agent)
    self.assertIn('"name": "echo_with_length"', users[0])
    self.assertEqual(json.loads(users[1]), {"message": "hello", "length": 5})
    self.assertLen(rewards, 2)  # max_steps ends the episode.

  def test_bad_actions_go_back_to_the_model(self):
    env = env_lib.OpenEnvTaskEnv(
        base_url=_url("echo"), max_steps=3, invalid_action_reward=-0.5
    )
    replies = iter([
        "no tool call",
        _call("delete_everything", {}),
        _call("echo_message", {"message": "ok"}),
    ])
    agent, rewards = _run(env, lambda messages: next(replies))
    users = _users(agent)
    self.assertStartsWith(users[1], "Invalid action: No <tool_call>")
    self.assertStartsWith(
        users[2], "Invalid action: Unknown function 'delete_everything'"
    )
    self.assertEqual(users[3], "ok")
    self.assertEqual(rewards, [-0.5, -0.5, 0.0])


class GuessTest(absltest.TestCase):

  def _env(self, seed, **kwargs):
    kwargs.setdefault("max_steps", 10)
    return env_lib.OpenEnvTaskEnv(
        {"seed": seed}, base_url=_url("guess"), max_sessions=4, **kwargs
    )

  def test_reward_and_done(self):
    _, rewards = _run(self._env(7), _binary_search_policy())
    self.assertEqual(rewards[-1], 1.0)
    self.assertEqual(sum(rewards), 1.0)
    self.assertLessEqual(len(rewards), 5)  # log2(20) guesses at most.

  def test_max_steps_truncates(self):
    env = self._env(7, max_steps=2)
    _, rewards = _run(env, lambda messages: _call("guess", {"number": 0}))
    self.assertEqual(rewards, [0.0, 0.0])

  def test_group_shares_a_task_and_pool_caps_sessions(self):
    """12 episodes at once, as one GRPO step runs them, on 4 sessions."""

    async def main():
      envs = [
          self._env(seed, group_id=seed) for seed in [1, 2, 3] for _ in range(4)
      ]
      results = await asyncio.gather(
          *[_run_episode(env, _binary_search_policy()) for env in envs]
      )
      return envs, results

    envs, results = asyncio.run(main())
    self.assertTrue(all(rewards[-1] == 1.0 for _, rewards in results))
    # All 4 sessions were busy at once, never more, and they were reused.
    pool = envs[0]._pool
    self.assertEqual(pool.peak_in_use, 4)
    self.assertEqual(pool.opened, 4)
    # Same seed, same hidden number, so the same number of guesses.
    lengths = collections.defaultdict(set)
    for env, (_, rewards) in zip(envs, results):
      lengths[env.extra_kwargs["group_id"]].add(len(rewards))
    self.assertTrue(all(len(v) == 1 for v in lengths.values()))

  def test_close_mid_episode_returns_the_session(self):
    env = self._env(1)
    env.reset()
    in_use = env._pool.in_use
    env.close()
    self.assertEqual(env._pool.in_use, in_use - 1)
    with self.assertRaisesRegex(RuntimeError, "before reset"):
      env.step(_call("guess", {"number": 1}))

  def test_session_errors_end_the_trajectory(self):
    env = self._env(1)
    env.reset()

    async def broken(*args, **kwargs):
      raise RuntimeError("Server error: gone (code: SESSION_ERROR)")

    env._client.step = broken
    with self.assertRaisesRegex(RuntimeError, "SESSION_ERROR"):
      env.step(_call("guess", {"number": 3}))
    client = env._client
    env.close()
    self.assertNotIn(client, env._pool._idle)  # Closed, not reused.
    self.assertIsNone(client._ws)


_RG_RESET = {"dataset_name": "leg_counting", "dataset_config": {}, "size": 50}


class ReasoningGymTest(absltest.TestCase):

  def _env(self, seed):
    return env_lib.OpenEnvTaskEnv(
        {"seed": seed},
        base_url=_url("reasoning_gym"),
        max_steps=1,
        reset_kwargs=_RG_RESET,
    )

  def _question(self, seed, reply):
    agent, rewards = _run(self._env(seed), lambda messages: reply)
    first = _users(agent)[0].split("\n\n")[-1]
    return json.loads(first)["question"], rewards

  def test_same_seed_same_question(self):
    question_a, rewards = self._question(11, _call("step", {"answer": "-1"}))
    question_b, _ = self._question(11, "-1")  # Raw text: single-string field.
    question_c, _ = self._question(12, "-1")
    self.assertEqual(question_a, question_b)
    self.assertNotEqual(question_a, question_c)
    self.assertEqual(rewards, [0.0])

  def test_correct_answer_scores_one(self):
    env = self._env(5)
    env.reset()
    _, _, done, info = env.step(_call("step", {"answer": "0"}))
    self.assertTrue(done)
    correct = info["openenv"]["observation"]["correct_answer"]
    env = self._env(5)
    env.reset()
    _, reward, done, _ = env.step(_call("step", {"answer": correct}))
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)


class MazeTest(absltest.TestCase):

  def test_registered_codec_reaches_exit(self):
    env = env_lib.OpenEnvTaskEnv(base_url=_url("maze"), max_steps=40)
    self.assertIsInstance(env.codec, maze_codec.MazeCodec)
    agent, rewards = _run(env, _bfs_policy)
    self.assertTrue(agent.trajectory.steps[-1].done)
    self.assertIn("Status: win", _last_user(agent.chat_completions))
    self.assertEqual(rewards[-1], 10.0)
    self.assertGreater(sum(rewards), 0)

  def test_default_codec_still_runs(self):
    """Without the maze codec the model sees only integers; the loop works."""
    env = env_lib.OpenEnvTaskEnv(
        base_url=_url("maze"), max_steps=3, codec=codecs.SchemaCodec()
    )
    replies = iter([
        _call("step", {"action": "up"}),
        _call("step", {"action": 3}),
        '{"action": 1}',
    ])
    agent, rewards = _run(env, lambda messages: next(replies))
    users = _users(agent)
    self.assertIn('"MazeAction"', users[0])
    self.assertStartsWith(
        users[1], "Invalid action: the environment rejected {'action': 'up'}"
    )
    self.assertEqual(json.loads(users[2])["current_position"], [1, 0])
    self.assertLen(rewards, 3)

  def test_out_of_range_action_keeps_the_session(self):
    env = env_lib.OpenEnvTaskEnv(
        base_url=_url("maze"), max_steps=5, codec=codecs.SchemaCodec()
    )
    env.reset()
    obs, _, done, _ = env.step(_call("step", {"action": 9}))
    self.assertStartsWith(obs, "Invalid action")
    self.assertFalse(done)
    obs, _, _, _ = env.step(_call("step", {"action": 3}))
    self.assertEqual(json.loads(obs)["current_position"], [1, 0])
    env.close()


if __name__ == "__main__":
  absltest.main()
