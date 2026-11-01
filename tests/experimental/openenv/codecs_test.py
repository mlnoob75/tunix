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

"""Tests for the OpenEnv codecs. No server needed."""

import json
import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized

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
from tunix.experimental.openenv import maze_codec
from tunix.experimental.openenv import spec as spec_lib

MCP_SPEC = spec_lib.EnvSpec(
    base_url="http://x",
    name="EchoEnvironment",
    kind="mcp",
    action_schema={},
    observation_schema={},
    reset_schema={},
    tools=(
        {
            "name": "echo_message",
            "description": "Echo back.",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        },
    ),
)

ANSWER_SPEC = spec_lib.EnvSpec(
    base_url="http://x",
    name="ReasoningGymEnvironment",
    kind="typed",
    action_schema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
    observation_schema={},
    reset_schema={},
)

INT_SPEC = spec_lib.EnvSpec(
    base_url="http://x",
    name="MazeEnvironment",
    kind="typed",
    action_schema={
        "type": "object",
        "properties": {"action": {"type": "integer"}},
        "required": ["action"],
    },
    observation_schema={},
    reset_schema={},
)


def _call(name, arguments):
  body = json.dumps({"name": name, "arguments": arguments})
  return f"<tool_call>{body}</tool_call>"


class ParseToolCallTest(parameterized.TestCase):

  def test_none_without_tags(self):
    self.assertIsNone(codecs.parse_tool_call("just text"))

  def test_first_call_wins(self):
    text = "think" + _call("a", {"x": 1}) + _call("b", {})
    self.assertEqual(
        codecs.parse_tool_call(text), {"name": "a", "arguments": {"x": 1}}
    )

  def test_string_arguments_are_decoded(self):
    text = '<tool_call>{"name": "a", "arguments": "{\\"x\\": 2}"}</tool_call>'
    self.assertEqual(codecs.parse_tool_call(text)["arguments"], {"x": 2})

  @parameterized.parameters(
      "{not json", '{"arguments": {}}', '{"name": "a", "arguments": [1]}'
  )
  def test_bad_bodies_raise(self, body):
    with self.assertRaises(codecs.ActionParseError):
      codecs.parse_tool_call(f"<tool_call>{body}</tool_call>")


class ToolCallCodecTest(absltest.TestCase):

  def test_instructions_list_every_tool(self):
    text = codecs.ToolCallCodec().instructions(MCP_SPEC)
    self.assertIn('"name": "echo_message"', text)
    self.assertIn("<tool_call>", text)

  def test_parse_builds_call_tool_payload(self):
    payload = codecs.ToolCallCodec().parse_action(
        _call("echo_message", {"message": "hi"}), MCP_SPEC
    )
    self.assertEqual(
        payload,
        {
            "type": "call_tool",
            "tool_name": "echo_message",
            "arguments": {"message": "hi"},
        },
    )

  def test_unknown_tool_and_missing_call_raise(self):
    codec = codecs.ToolCallCodec()
    with self.assertRaisesRegex(codecs.ActionParseError, "Unknown function"):
      codec.parse_action(_call("rm", {}), MCP_SPEC)
    with self.assertRaisesRegex(codecs.ActionParseError, "No <tool_call>"):
      codec.parse_action("hello", MCP_SPEC)

  def test_render_prefers_data_then_error_then_text(self):
    codec = codecs.ToolCallCodec()
    obs = {"tool_name": "t", "result": {"data": {"a": 1}, "content": []}}
    self.assertEqual(codec.render_observation(obs, MCP_SPEC), '{"a": 1}')
    obs = {
        "tool_name": "t",
        "result": None,
        "error": {"error_type": "tool_not_found", "message": "no"},
    }
    self.assertEqual(
        codec.render_observation(obs, MCP_SPEC),
        "Tool error (tool_not_found): no",
    )
    obs = {
        "tool_name": "t",
        "result": {"content": [{"type": "text", "text": "x"}]},
    }
    self.assertEqual(codec.render_observation(obs, MCP_SPEC), "x")


class SchemaCodecTest(absltest.TestCase):

  def test_tool_call(self):
    self.assertEqual(
        codecs.SchemaCodec().parse_action(
            _call("step", {"action": 2}), INT_SPEC
        ),
        {"action": 2},
    )

  def test_bare_json_needs_action_keys(self):
    codec = codecs.SchemaCodec()
    self.assertEqual(
        codec.parse_action('I pick {"action": 3}', INT_SPEC), {"action": 3}
    )
    with self.assertRaises(codecs.ActionParseError):
      codec.parse_action('my plan is {"goal": "exit"}', INT_SPEC)

  def test_single_string_field_takes_raw_text(self):
    self.assertEqual(
        codecs.SchemaCodec().parse_action(" 42 ", ANSWER_SPEC), {"answer": "42"}
    )

  def test_wrong_function_name_raises(self):
    with self.assertRaisesRegex(codecs.ActionParseError, "Call 'step'"):
      codecs.SchemaCodec().parse_action(_call("move", {}), INT_SPEC)

  def test_render_hides_none_metadata_and_hidden_fields(self):
    obs = {
        "question": "q",
        "score": None,
        "correct_answer": "7",
        "metadata": {"debug": 1},
    }
    codec = codecs.SchemaCodec(hide_fields=["correct_answer"])
    self.assertEqual(
        codec.render_observation(obs, ANSWER_SPEC), '{"question": "q"}'
    )
    codec = codecs.SchemaCodec(show_metadata=True)
    self.assertIn('"metadata"', codec.render_observation(obs, ANSWER_SPEC))


class RegistryTest(absltest.TestCase):

  def test_defaults_by_kind(self):
    self.assertIsInstance(codecs.codec_for(MCP_SPEC), codecs.ToolCallCodec)
    self.assertIsInstance(codecs.codec_for(ANSWER_SPEC), codecs.SchemaCodec)

  def test_registered_codec_wins(self):
    self.assertIsInstance(codecs.codec_for(INT_SPEC), maze_codec.MazeCodec)


class MazeCodecTest(absltest.TestCase):

  def test_parse_last_move(self):
    codec = maze_codec.MazeCodec()
    self.assertEqual(
        codec.parse_action("<move>up</move> no, <move>Right</move>", INT_SPEC),
        {"action": 3},
    )
    with self.assertRaises(codecs.ActionParseError):
      codec.parse_action("go right", INT_SPEC)

  def test_render_grid(self):
    obs = {
        "current_position": [0, 0],
        "legal_actions": [1, 3],
        "metadata": {
            "maze": [[0, 1], [0, 0]],
            "exit_cell": [1, 1],
            "status": "playing",
        },
    }
    self.assertEqual(
        maze_codec.MazeCodec().render_observation(obs, INT_SPEC),
        "A#\n.E\nStatus: playing. Legal moves: down, right.",
    )


if __name__ == "__main__":
  absltest.main()
