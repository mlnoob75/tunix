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

"""A bounded pool of WebSocket sessions per OpenEnv server.

Tunix builds one env per example per group member, so a GRPO batch of 32 with
a group size of 8 asks for 256 episodes at once. OpenEnv servers cap
concurrent sessions (`MAX_CONCURRENT_ENVS`, 8 by default in the reference
servers). The pool holds a session for one episode, from `reset()` to
`close()`, makes extra episodes wait, and reuses connected sessions.
"""

import asyncio
import threading
from typing import Dict, List, Optional, Tuple

from openenv.core.generic_client import GenericEnvClient
from tunix.experimental.openenv import _loop


class SessionPool:
  """Sessions to one server. Use only from the shared loop (see `_loop`)."""

  def __init__(
      self,
      base_url: str,
      max_sessions: int,
      headers: Optional[Dict[str, str]] = None,
      message_timeout_s: float = 60.0,
  ):
    self.base_url = base_url
    self.max_sessions = max_sessions
    self._headers = headers
    self._message_timeout_s = message_timeout_s
    self._slots = asyncio.Semaphore(max_sessions)
    self._idle: List[GenericEnvClient] = []
    self.opened = 0  # Sessions ever connected. For tests and logs.
    self.in_use = 0
    self.peak_in_use = 0

  async def acquire(self) -> GenericEnvClient:
    """Waits for a free slot, then returns an idle or new session."""
    await self._slots.acquire()
    try:
      if self._idle:
        client = self._idle.pop()
      else:
        client = GenericEnvClient(
            base_url=self.base_url,
            headers=self._headers,
            message_timeout_s=self._message_timeout_s,
        )
        await client.connect()
        self.opened += 1
    except BaseException:
      self._slots.release()
      raise
    self.in_use += 1
    self.peak_in_use = max(self.peak_in_use, self.in_use)
    return client

  async def release(self, client: GenericEnvClient, healthy: bool = True):
    """Returns a session. An unhealthy one is closed instead of reused."""
    self.in_use -= 1
    try:
      if healthy:
        self._idle.append(client)
      else:
        await client.close()
    finally:
      self._slots.release()

  async def close(self):
    """Closes every idle session."""
    idle, self._idle = self._idle, []
    for client in idle:
      await client.close()


_pools: Dict[Tuple[str, int], SessionPool] = {}
_pools_lock = threading.Lock()


def get_pool(
    base_url: str,
    max_sessions: int,
    headers: Optional[Dict[str, str]] = None,
    message_timeout_s: float = 60.0,
) -> SessionPool:
  """The process-wide pool for this server and size."""
  key = (base_url.rstrip("/"), max_sessions)
  with _pools_lock:
    if key not in _pools:

      async def make():
        # The semaphore must be created on the shared loop.
        return SessionPool(key[0], max_sessions, headers, message_timeout_s)

      _pools[key] = _loop.run(make())
    return _pools[key]


def close_all():
  """Closes every idle session. Call at the end of training."""
  with _pools_lock:
    pools = list(_pools.values())
    _pools.clear()
  for pool in pools:
    _loop.run(pool.close())
