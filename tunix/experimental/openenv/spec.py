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

"""Finds out what an OpenEnv server accepts, from the server alone.

Every OpenEnv server publishes `/schema` (action, observation and reset JSON
schemas) and `/metadata` (environment name). MCP environments also list their
tools through a `list_tools` step. That is enough to drive an environment the
adapter has never seen, without installing the environment's own package.
"""

import dataclasses
import json
import threading
from typing import Any, Dict, Optional, Tuple
import urllib.request

from openenv.core.generic_client import GenericEnvClient
from tunix.experimental.openenv import _loop

KIND_MCP = "mcp"
KIND_TYPED = "typed"


@dataclasses.dataclass(frozen=True)
class EnvSpec:
  """What one OpenEnv server accepts and returns.

  Attributes:
    base_url: Server URL.
    name: Environment name from `/metadata` (for example `MazeEnvironment`).
    kind: `"mcp"` for tool environments, `"typed"` for reset/step ones.
    action_schema: JSON schema of the action, without the `metadata` field.
    observation_schema: JSON schema of the step observation.
    reset_schema: JSON schema of the `reset()` keyword arguments.
    tools: For MCP environments, each tool as `{name, description,
      input_schema}`. Empty for typed environments.
  """

  base_url: str
  name: str
  kind: str
  action_schema: Dict[str, Any]
  observation_schema: Dict[str, Any]
  reset_schema: Dict[str, Any]
  tools: Tuple[Dict[str, Any], ...] = ()

  @property
  def reset_params(self) -> Tuple[str, ...]:
    return tuple(self.reset_schema.get("properties", {}))


def _get_json(url: str, headers: Optional[Dict[str, str]], timeout: float):
  request = urllib.request.Request(url, headers=headers or {})
  with urllib.request.urlopen(request, timeout=timeout) as response:
    return json.load(response)


def _strip_metadata(schema: Dict[str, Any]) -> Dict[str, Any]:
  """Drops the `metadata` field every OpenEnv action inherits."""
  schema = dict(schema)
  properties = dict(schema.get("properties", {}))
  properties.pop("metadata", None)
  schema["properties"] = properties
  if "required" in schema:
    schema["required"] = [r for r in schema["required"] if r != "metadata"]
  return schema


def _is_mcp(action_schema: Dict[str, Any]) -> bool:
  properties = action_schema.get("properties", {})
  action_type = properties.get("type", {})
  return "tool_name" in properties and (
      action_type.get("default") == "call_tool"
      or action_type.get("const") == "call_tool"
  )


async def _list_tools(
    base_url: str, headers: Optional[Dict[str, str]]
) -> Tuple[Dict[str, Any], ...]:
  """Opens a short session and asks an MCP server for its tools."""
  client = GenericEnvClient(base_url=base_url, headers=headers)
  try:
    await client.connect()
    await client.reset()
    result = await client.step({"type": "list_tools"})
    return tuple(result.observation.get("tools", []))
  finally:
    await client.close()


def discover(
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> EnvSpec:
  """Reads the schemas (and, for MCP servers, the tools) of a server."""
  base_url = base_url.rstrip("/")
  schemas = _get_json(f"{base_url}/schema", headers, timeout)
  try:
    name = _get_json(f"{base_url}/metadata", headers, timeout).get("name", "")
  except OSError:
    name = ""
  action_schema = _strip_metadata(schemas["action"])
  kind = KIND_MCP if _is_mcp(action_schema) else KIND_TYPED
  tools = ()
  if kind == KIND_MCP:
    tools = _loop.run(_list_tools(base_url, headers), timeout)
  return EnvSpec(
      base_url=base_url,
      name=name,
      kind=kind,
      action_schema=action_schema,
      observation_schema=schemas.get("observation", {}),
      reset_schema=schemas.get("reset", {}),
      tools=tools,
  )


_cache: Dict[str, EnvSpec] = {}
_cache_lock = threading.Lock()


def discover_cached(
    base_url: str, headers: Optional[Dict[str, str]] = None
) -> EnvSpec:
  """`discover()`, once per server per process."""
  key = base_url.rstrip("/")
  with _cache_lock:
    if key not in _cache:
      _cache[key] = discover(key, headers)
    return _cache[key]
