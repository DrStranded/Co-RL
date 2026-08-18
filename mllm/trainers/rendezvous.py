"""File-based rendezvous for exchanging pseudo-labels between two co_rl groups.

Each group runs in its own accelerate world (different CUDA devices, different
master port). They share nothing at the `torch.distributed` level; cross-group
communication is done exclusively by writing and polling JSON files in a
shared directory on the filesystem.

Protocol per exchange (producer/consumer):
    1. Write own payload atomically to  <dir>/<mode>_<counter>_<me>.json  (tmp + rename)
    2. Poll for peer's file              <dir>/<mode>_<counter>_<peer>.json
    3. Read peer payload
    4. Delete the *peer's* file (I consumed it). Never delete my own file -
       that is the peer's responsibility after they read it.

This producer/consumer split prevents a race where a fast peer deletes its own
file before we poll for it. File existence here means "produced and not yet
consumed"; once consumed, the consumer deletes it.

Only one rank per group should call `exchange()` (typically `accelerator.is_main_process`).
The caller is responsible for broadcasting the returned payload to the rest of the group.
"""

import json
import os
import time
from pathlib import Path


class Rendezvous:
    """
    Args:
        rendezvous_dir (`str`):
            Directory shared between the two groups. On a single machine, `/tmp/...`
            or a path inside the experiment's output dir is fine. On multi-node, this
            must live on a shared filesystem (NFS, etc.).
        my_group_name (`str`):
            `'A'` or `'B'`.
        poll_interval (`float`, *optional*, defaults to `0.05`):
            Seconds to sleep between polls while waiting for the peer's file.
        timeout (`float`, *optional*, defaults to `3600.0`):
            Maximum seconds to wait for the peer before raising `TimeoutError`.
            Catches the case where the peer process has silently crashed.
    """

    def __init__(
        self,
        rendezvous_dir: str,
        my_group_name: str,
        poll_interval: float = 0.05,
        timeout: float = 3600.0,
    ):
        assert my_group_name in ("A", "B"), f"my_group_name must be 'A' or 'B', got {my_group_name!r}"
        self.dir = Path(rendezvous_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.me = my_group_name
        self.peer = "B" if my_group_name == "A" else "A"
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _path(self, mode: str, counter: int, group: str) -> Path:
        return self.dir / f"{mode}_{counter}_{group}.json"

    def exchange(self, mode: str, counter: int, payload: list) -> list:
        """Write `payload` and block until the peer's payload for the same (mode, counter) appears.

        Args:
            mode (`str`):
                `'train'` or `'eval'`. Separates keys so switching between train and eval
                evaluation does not misalign the two groups' counters.
            counter (`int`):
                Monotonically increasing per (mode) on each group. Must match between groups.
            payload (`list`):
                JSON-serializable list (typically pseudo-labels as strings). Sent as-is.

        Returns:
            `list`: the peer group's payload for this (mode, counter).
        """
        my_path = self._path(mode, counter, self.me)
        peer_path = self._path(mode, counter, self.peer)

        # Atomic write: write to tmp + rename. Prevents the peer from reading a
        # half-written file on platforms where rename is atomic (Linux is).
        tmp = my_path.with_suffix(my_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, my_path)

        # Poll for peer's file. Peer may arrive before or after us.
        start = time.time()
        while not peer_path.exists():
            if time.time() - start > self.timeout:
                raise TimeoutError(
                    f"[rendezvous {self.me}] peer {self.peer} did not write "
                    f"{peer_path.name} within {self.timeout}s, peer likely crashed."
                )
            time.sleep(self.poll_interval)

        # Read with retry. On rare occasions (very fast poll + slow fs flush), we can
        # see the path exist but read a partial payload. Atomic rename makes this
        # unlikely, but we guard against it anyway.
        peer_payload = None
        for _ in range(20):
            try:
                peer_payload = json.loads(peer_path.read_text())
                break
            except (json.JSONDecodeError, FileNotFoundError):
                time.sleep(self.poll_interval)
        if peer_payload is None:
            raise RuntimeError(f"[rendezvous {self.me}] failed to parse {peer_path}")

        # I consumed peer_path; delete it. Never delete my_path, that is peer's job.
        try:
            peer_path.unlink()
        except FileNotFoundError:
            pass

        return peer_payload
