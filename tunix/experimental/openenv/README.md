# OpenEnv environments for agentic RL (experimental)

`OpenEnvTaskEnv` runs any text environment served by
[OpenEnv](https://github.com/huggingface/OpenEnv) in Tunix's agentic RL learners
(`AgenticGRPOLearner` and others that take `env_class`).

```python
from tunix.experimental.openenv.env import OpenEnvTaskEnv
from tunix.rl.agentic.agents.model_agent import ModelAgent

learner = GRPOLearner(
    ...,
    agent_class=ModelAgent,
    env_class=OpenEnvTaskEnv,
    env_kwargs={
        "base_url": "http://localhost:8000",
        "max_steps": 8,
        "max_sessions": 8,
    },
)
```

Install with `pip install "google-tunix[openenv]"`.

Status: tested on CPU against four local OpenEnv servers with scripted policies.
**Not yet run in a real training job.** See
[examples/openenv/train_grpo.py](../examples/openenv/train_grpo.py).

## How it works

The adapter learns everything it needs from the server, so it does not need the
environment's own package:

| Server gives                                      | Used for                                              |
| ------------------------------------------------- | ----------------------------------------------------- |
| `/schema` (action, observation and reset schemas) | Action format, and which dataset keys go to `reset()` |
| `/metadata` name                                  | Picking a registered per-environment codec            |
| `list_tools` step (MCP servers)                   | Tool list shown to the model                          |

A **codec** ([codecs.py](codecs.py)) translates between model text and the
environment:

- `ToolCallCodec` (MCP environments): every MCP tool becomes a function, in the
  `<tool_call>` format of Tunix's Qwen tool parser.
- `SchemaCodec` (typed environments): the action schema becomes one `step`
  function. It also accepts a bare JSON action, and raw text when the action is
  a single string field (for example `{"answer": ...}`).
- A per-environment codec, registered with
  `@codecs.register_codec("<metadata name>")`, for environments whose default
  text is poor. [maze_codec.py](maze_codec.py) is the example: by default the
  model sees only `{"action": <int>}` and `legal_actions: [3]`. The codec draws
  the grid and uses `<move>right</move>`.

Episodes:

- Observations are strings, so the stock `ModelAgent` works. The first
  observation starts with the codec's action instructions.
- Each dataset example becomes the `reset()` arguments: its `reset_kwargs` dict,
  or else the keys that the reset schema accepts (usually `seed`). The same
  example goes to every member of a GRPO group, so they share one task.
- The step reward is the server's reward (`None` counts as 0). Tunix sums the
  step rewards.
- A bad reply (no parseable action, an unknown tool, or a server
  `VALIDATION_ERROR` or `EXECUTION_ERROR`) goes back to the model as
  `Invalid action: ...`, with `invalid_action_reward`. It counts as a step.
  Other server errors end the trajectory.

Concurrency:

- Tunix calls `reset()` and `step()` from executor threads. All OpenEnv clients
  run on one shared background event loop ([\_loop.py](_loop.py)).
- A session pool per server ([pool.py](pool.py)) caps open sessions at
  `max_sessions` and reuses them. Without it, one GRPO step overruns the server:
  with 12 episodes on a 4-session server, 8 fail with `CAPACITY_REACHED`. With
  the pool, all 12 finish.

## Limits

- No image observations (Atari, dm_control, CARLA, Unity, BrowserGym
  screenshots).
- MCP tools are listed once per server, so tool lists that change per episode
  are not supported.
- Reset arguments that are not in the server's reset schema (for example
  reasoning_gym's `dataset_name`) are not taken from dataset examples. Put them
  in `env_kwargs["reset_kwargs"]` or in a per-example `reset_kwargs` dict.
- OpenEnv's harness API, where the environment runs the whole episode, is not
  used. Tunix keeps control of the loop.
- Codec objects in `env_kwargs` work with the in-process learners. The `dist_rl`
  stack, which can serialize `env_kwargs`, is not tested.

## Tests

The tests in `tests/experimental/openenv/` skip when `openenv` is not installed.
The end-to-end tests start a server for each environment package that is
installed: OpenEnv's `echo_env` (MCP), `reasoning_gym_env` (typed, one turn) and
`maze_env` (typed, multi-turn), and `guess_env.py` (a small MCP environment with
rewards and a 4-session limit).

```bash
pip install -e path/to/OpenEnv -e path/to/OpenEnv/envs/echo_env \
    -e path/to/OpenEnv/envs/reasoning_gym_env -e path/to/OpenEnv/envs/maze_env
python -m pytest tests/experimental/openenv/
```
