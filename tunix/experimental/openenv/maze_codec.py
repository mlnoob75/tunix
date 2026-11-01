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

"""Codec for OpenEnv's maze_env: an example of a per-environment codec.

The default `SchemaCodec` works for the maze, but the model sees only
`{"action": <integer>}` and `legal_actions: [3]`. It cannot know that 3 means
"right", and the grid is hidden in the observation metadata. This codec draws
the grid and uses direction words.

Import this module to register the codec for `MazeEnvironment`.
"""

import re

from tunix.experimental.openenv import codecs

_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
_IDS = {name: action for action, name in _NAMES.items()}
_MOVE_RE = re.compile(r"<move>\s*(up|down|left|right)\s*</move>", re.IGNORECASE)


class MazeCodec(codecs.Codec):
  """Grid as text, moves as `<move>up</move>`."""

  def instructions(self, spec):
    return (
        "You are in a maze. '#' is a wall, '.' is free, 'A' is you and 'E' is"
        " the exit. Each move costs a little, walking into a wall costs more,"
        " and reaching the exit gives a large reward. Reply with one move:"
        " <move>up</move>, <move>down</move>, <move>left</move> or"
        " <move>right</move>."
    )

  def parse_action(self, text, spec):
    matches = _MOVE_RE.findall(text)
    if not matches:
      raise codecs.ActionParseError(
          "Reply with one move, for example <move>right</move>."
      )
    return {"action": _IDS[matches[-1].lower()]}

  def render_observation(self, observation, spec):
    metadata = observation.get("metadata") or {}
    maze = metadata.get("maze")
    if not maze:
      return codecs.SchemaCodec().render_observation(observation, spec)
    col, row = observation.get("current_position") or [None, None]
    exit_col, exit_row = metadata.get("exit_cell") or [None, None]
    lines = []
    for r, values in enumerate(maze):
      cells = []
      for c, wall in enumerate(values):
        if (c, r) == (col, row):
          cells.append("A")
        elif (c, r) == (exit_col, exit_row):
          cells.append("E")
        else:
          cells.append("#" if wall else ".")
      lines.append("".join(cells))
    legal = [_NAMES[a] for a in observation.get("legal_actions") or []]
    status = metadata.get("status", "playing")
    return (
        "\n".join(lines)
        + f"\nStatus: {status}. Legal moves: {', '.join(legal) or 'none'}."
    )


@codecs.register_codec("MazeEnvironment")
def _make():
  return MazeCodec()
