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

"""One background event loop shared by every adapter in the process.

Tunix calls `env.reset()` and `env.step()` from executor threads. The OpenEnv
clients are async and bind their WebSocket to the loop that opened it. So all
client calls go through this one loop thread, instead of one loop per env.
"""

import asyncio
import threading
from typing import Any, Coroutine, Optional

_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_loop() -> asyncio.AbstractEventLoop:
  global _loop
  with _lock:
    if _loop is None or _loop.is_closed():
      _loop = asyncio.new_event_loop()
      thread = threading.Thread(
          target=_loop.run_forever, name="tunix-openenv-loop", daemon=True
      )
      thread.start()
    return _loop


def run(coro: Coroutine[Any, Any, Any], timeout: Optional[float] = None) -> Any:
  """Runs `coro` on the shared loop and blocks the calling thread for it.

  The timeout is applied inside the loop with `asyncio.wait_for`, so a coroutine
  that times out is cancelled there and cannot finish later unobserved (for
  example, take a pool slot after its caller gave up).
  """
  if timeout is not None:
    coro = asyncio.wait_for(coro, timeout)
  return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()
