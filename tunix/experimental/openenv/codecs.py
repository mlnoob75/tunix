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

"""Translation between model text and OpenEnv actions and observations.

A codec does three things:

* `instructions()`: tells the model which actions it can take.
* `parse_action()`: turns the model's reply into an OpenEnv step payload.
* `render_observation()`: turns an OpenEnv observation into text.

Two default codecs cover every text environment:

* `ToolCallCodec` for MCP environments: each MCP tool is a function.
* `SchemaCodec` for typed environments: the action schema from `/schema` is a
  single `step` function.

Both use the `<tool_call>` format of Tunix's Qwen tool parser, so models
that already know that format need no new prompt style. An environment whose
default text is poor gets its own codec through `register_codec()`.
"""

import abc
import json
import re
from typing import Any, Callable, Dict, Optional, Sequence

from tunix.experimental.openenv import spec as spec_lib

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ActionParseError(ValueError):
  """The model's reply is not a valid action. The message goes to the model."""


def _tools_prompt(functions: Sequence[Dict[str, Any]]) -> str:
  tools = "\n".join(
      json.dumps({"type": "function", "function": f}) for f in functions
  )
  return (
      "You are provided with function signatures within <tools></tools> XML"
      f" tags:\n<tools>\n{tools}\n</tools>\n\nMake exactly one function call"
      " per turn. Return a json object with function name and arguments within"
      ' <tool_call></tool_call> XML tags:\n<tool_call>\n{"name":'
      ' "<function-name>", "arguments": <args-json-object>}\n</tool_call>'
  )


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
  """Returns the first `{"name", "arguments"}` call in `text`, or None."""
  match = _TOOL_CALL_RE.search(text)
  if match is None:
    return None
  try:
    call = json.loads(match.group(1))
  except json.JSONDecodeError as e:
    raise ActionParseError(
        f"The <tool_call> body is not valid JSON: {e}"
    ) from e
  if not isinstance(call, dict) or "name" not in call:
    raise ActionParseError(
        'The <tool_call> body must be {"name": ..., "arguments": {...}}.'
    )
  arguments = call.get("arguments") or {}
  if isinstance(arguments, str):
    try:
      arguments = json.loads(arguments)
    except json.JSONDecodeError as e:
      raise ActionParseError(f"The arguments are not valid JSON: {e}") from e
  if not isinstance(arguments, dict):
    raise ActionParseError("The arguments must be a JSON object.")
  return {"name": call["name"], "arguments": arguments}


def _compact(value: Any) -> str:
  if isinstance(value, str):
    return value
  return json.dumps(value, default=str)


class Codec(abc.ABC):
  """Translates between model text and one OpenEnv environment."""

  @abc.abstractmethod
  def instructions(self, spec: spec_lib.EnvSpec) -> str:
    """Text that tells the model how to act.

    Sent once, before the first observation.
    """

  @abc.abstractmethod
  def parse_action(self, text: str, spec: spec_lib.EnvSpec) -> Dict[str, Any]:
    """Returns the step payload for `text`. Raises `ActionParseError`."""

  @abc.abstractmethod
  def render_observation(
      self, observation: Dict[str, Any], spec: spec_lib.EnvSpec
  ) -> str:
    """Returns the text the model sees for `observation`."""


class ToolCallCodec(Codec):
  """Default codec for MCP environments."""

  def instructions(self, spec):
    functions = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object"}),
        }
        for t in spec.tools
    ]
    return _tools_prompt(functions)

  def parse_action(self, text, spec):
    call = parse_tool_call(text)
    if call is None:
      raise ActionParseError(
          "No <tool_call> found. Call one of the provided functions."
      )
    names = [t["name"] for t in spec.tools]
    if call["name"] not in names:
      raise ActionParseError(
          f"Unknown function {call['name']!r}. Use one of {names}."
      )
    return {
        "type": "call_tool",
        "tool_name": call["name"],
        "arguments": call["arguments"],
    }

  def render_observation(self, observation, spec):
    error = observation.get("error")
    if error:
      return (
          f"Tool error ({error.get('error_type', 'error')}):"
          f" {error.get('message', '')}"
      )
    if "tool_name" not in observation:
      # A reset observation: show what the environment says about itself.
      return _compact(observation.get("metadata") or {})
    result = observation.get("result")
    if isinstance(result, dict):
      if result.get("data") is not None:
        return _compact(result["data"])
      if result.get("structured_content") is not None:
        return _compact(result["structured_content"])
      texts = [
          c.get("text", "")
          for c in result.get("content") or []
          if c.get("type") == "text"
      ]
      return "\n".join(texts)
    return _compact(result)


class SchemaCodec(Codec):
  """Default codec for typed environments.

  Args:
    hide_fields: Observation fields the model must not see (for example an
      answer the environment returns for logging).
    show_metadata: Include the observation's `metadata` dict. Off by default,
      because servers often put debug data there.
  """

  def __init__(
      self, hide_fields: Sequence[str] = (), show_metadata: bool = False
  ):
    self.hide_fields = set(hide_fields)
    self.show_metadata = show_metadata

  def _single_string_field(self, spec) -> Optional[str]:
    properties = spec.action_schema.get("properties", {})
    required = spec.action_schema.get("required", [])
    if (
        len(required) == 1
        and properties.get(required[0], {}).get("type") == "string"
    ):
      return required[0]
    return None

  def instructions(self, spec):
    function = {
        "name": "step",
        "description": f"Take one action in the {spec.name or 'environment'}.",
        "parameters": spec.action_schema,
    }
    return _tools_prompt([function])

  def parse_action(self, text, spec):
    call = parse_tool_call(text)
    if call is not None:
      if call["name"] != "step":
        raise ActionParseError(
            f"Unknown function {call['name']!r}. Call 'step'."
        )
      return call["arguments"]
    # A bare JSON action is accepted only if its keys are action fields, so
    # stray braces in the model's reasoning do not become actions.
    match = _JSON_OBJECT_RE.search(text)
    if match is not None:
      try:
        value = json.loads(match.group(0))
      except json.JSONDecodeError:
        value = None
      properties = spec.action_schema.get("properties", {})
      if isinstance(value, dict) and value and set(value) <= set(properties):
        return value
    field = self._single_string_field(spec)
    if field is not None and text.strip():
      return {field: text.strip()}
    raise ActionParseError("No <tool_call> found. Call the 'step' function.")

  def render_observation(self, observation, spec):
    shown = {
        k: v
        for k, v in observation.items()
        if v is not None
        and k not in self.hide_fields
        and (k != "metadata" or (self.show_metadata and v))
    }
    return _compact(shown)


_registry: Dict[str, Callable[[], Codec]] = {}


def register_codec(
    env_name: str,
) -> Callable[[Callable[[], Codec]], Callable[[], Codec]]:
  """Registers a codec factory for the env with this `/metadata` name."""

  def decorator(factory):
    _registry[env_name] = factory
    return factory

  return decorator


def codec_for(spec: spec_lib.EnvSpec) -> Codec:
  """The registered codec for `spec.name`, else the default for its kind."""
  if spec.name in _registry:
    return _registry[spec.name]()
  if spec.kind == spec_lib.KIND_MCP:
    return ToolCallCodec()
  return SchemaCodec()
