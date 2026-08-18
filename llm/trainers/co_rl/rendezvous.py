"""File-based rendezvous for exchanging pseudo-labels between co_rl groups.

Each group runs in its own accelerate world (different CUDA devices, different
master port). They share nothing at the `torch.distributed` level; cross-group
communication is done exclusively by writing and polling JSON files in a
shared directory on the filesystem.

Two protocols:

  - `exchange()`, 2-way (legacy). One outgoing file per call:
        <dir>/<mode>_<counter>_<me>.json
    Consumer deletes peer's file after read.

  - `exchange_n_way()`, N-way (N ≥ 2). One outgoing file per peer:
        <dir>/<mode>_<counter>_from-<me>_to-<peer>.json
    Each consumer only deletes its OWN incoming file, no cross-consumer race.
    This is the protocol used by 3+ way co_rl (e.g. N=3 Qwen x Llama x Gemma).

The N-way protocol is a strict superset of 2-way; we keep `exchange()` for
backward compatibility with existing 2-way scripts.

Only one rank per group should call exchange* (typically `accelerator.is_main_process`).
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
            Directory shared between the groups. On a single machine, `/tmp/...`
            or a path inside the experiment's output dir is fine. On multi-node, this
            must live on a shared filesystem (NFS, etc.).
        my_group_name (`str`):
            Group identifier, e.g. `'A'`, `'B'`, `'C'`, ...
        poll_interval (`float`, *optional*, defaults to `0.05`):
            Seconds to sleep between polls while waiting for the peer's file.
        timeout (`float`, *optional*, defaults to `3600.0`):
            Maximum seconds to wait for the peer before raising `TimeoutError`.
            Catches the case where the peer process has silently crashed.
        peers (`list[str]`, *optional*):
            Names of peer groups. If None, defaults to the legacy 2-way pair
            (A↔B). For N-way (N≥3) you must pass peers explicitly.
    """

    def __init__(
        self,
        rendezvous_dir: str,
        my_group_name: str,
        poll_interval: float = 0.05,
        timeout: float = 3600.0,
        peers: list[str] | None = None,
    ):
        self.dir = Path(rendezvous_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.me = my_group_name
        self.poll_interval = poll_interval
        self.timeout = timeout

        if peers is None:
            # Legacy 2-way default, assumes group names in {'A','B'}.
            assert my_group_name in ("A", "B"), (
                f"my_group_name must be 'A' or 'B' for legacy 2-way default; "
                f"for N-way pass `peers` explicitly. Got {my_group_name!r}."
            )
            self.peer = "B" if my_group_name == "A" else "A"
            self.peers = [self.peer]
        else:
            assert my_group_name not in peers, (
                f"my_group_name {my_group_name!r} should not appear in peers={peers!r}"
            )
            assert len(peers) == len(set(peers)), f"duplicate peers: {peers!r}"
            self.peers = list(peers)
            # Legacy `.peer` attr only valid in 2-way case; keep for back-compat.
            self.peer = self.peers[0] if len(self.peers) == 1 else None

    def _path(self, mode: str, counter: int, group: str) -> Path:
        """Legacy 2-way path layout. Used only by `exchange()`."""
        return self.dir / f"{mode}_{counter}_{group}.json"

    def _path_directed(self, mode: str, counter: int, src: str, dst: str) -> Path:
        """N-way path layout. Used by `exchange_n_way()`. One file per (src, dst)."""
        return self.dir / f"{mode}_{counter}_from-{src}_to-{dst}.json"

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

    def exchange_n_way(self, mode: str, counter: int, payload: list) -> dict[str, list]:
        """N-way variant: send `payload` to each of `self.peers` and collect their payloads.

        Per-call file layout (1 file per directed edge):
            outgoing[peer] = <dir>/<mode>_<counter>_from-<me>_to-<peer>.json
            incoming[peer] = <dir>/<mode>_<counter>_from-<peer>_to-<me>.json

        Each consumer deletes only its own incoming file, no race between
        consumers on a shared producer file (unlike the 2-way `exchange()`).

        Args:
            mode (`str`): `'train'` or `'eval'`.
            counter (`int`): Monotonically increasing per mode, must match across groups.
            payload (`list`): JSON-serializable list, sent identically to every peer.

        Returns:
            `dict[str, list]`: maps peer name -> that peer's payload. Order matches
            `self.peers`.
        """
        if not self.peers:
            raise RuntimeError(
                f"exchange_n_way called but rendezvous has no peers configured"
            )

        # ---- 1. Write N-1 outgoing files (one per peer) ----
        outgoing_payload = json.dumps(payload)
        for peer in self.peers:
            out_path = self._path_directed(mode, counter, self.me, peer)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_text(outgoing_payload)
            os.replace(tmp, out_path)

        # ---- 2. Poll for N-1 incoming files (one per peer) ----
        incoming_paths = {
            peer: self._path_directed(mode, counter, peer, self.me)
            for peer in self.peers
        }
        start = time.time()
        pending = dict(incoming_paths)
        while pending:
            done = []
            for peer, path in pending.items():
                if path.exists():
                    done.append(peer)
            for peer in done:
                del pending[peer]
            if pending:
                if time.time() - start > self.timeout:
                    missing = ", ".join(sorted(pending))
                    raise TimeoutError(
                        f"[rendezvous {self.me}] peers {{{missing}}} did not write "
                        f"their {mode}_{counter} payloads within {self.timeout}s, "
                        f"some peer likely crashed."
                    )
                time.sleep(self.poll_interval)

        # ---- 3. Read each incoming file with retry, then delete my own copy ----
        peer_payloads: dict[str, list] = {}
        for peer in self.peers:
            in_path = incoming_paths[peer]
            payload_in = None
            for _ in range(20):
                try:
                    payload_in = json.loads(in_path.read_text())
                    break
                except (json.JSONDecodeError, FileNotFoundError):
                    time.sleep(self.poll_interval)
            if payload_in is None:
                raise RuntimeError(
                    f"[rendezvous {self.me}] failed to parse incoming from {peer}: {in_path}"
                )
            peer_payloads[peer] = payload_in
            # I am the only consumer of this directed-edge file; delete it.
            try:
                in_path.unlink()
            except FileNotFoundError:
                pass

        return peer_payloads
