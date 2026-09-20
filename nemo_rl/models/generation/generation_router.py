# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""A NeMo-RL-owned HTTP router in front of the vLLM generation fleet.

NeMo-Gym picks a policy endpoint by static round-robin over a list fixed at process
start, with no health input and no failover. A dead vLLM endpoint therefore keeps
receiving roughly 1/N of new rollouts for the rest of the run, and Gym retries a refused
connection in an uncapped loop with no HTTP timeout.

Rather than change Gym, hand it a single URL that NeMo-RL owns. Gym's
``VLLMModelConfig.base_url`` accepts one string, so its round-robin becomes a no-op and
the routing decision moves here, next to the fleet health that already knows which shards
are serving.

Two properties make this safe to put in Gym's critical path:

* **The URL never changes.** The port is reserved once and passed in, so Ray recreating a
  restarted actor rebinds the same address. Gym is never reconfigured and never has to
  fail over -- which matters because failing over is exactly what it cannot do.
* **Every piece of state is built in __init__.** A restarted actor is immediately usable.
  This is the deliberate inverse of the NemoGym mistake, where the servers were started
  from a separate ``_spinup`` that Ray never re-runs.

Deliberately *not* a redirect. Handing Gym a 307 would put its socket back on a vLLM
endpoint directly, so a backend dying mid-request would drop it into the same uncapped
retry loop this exists to avoid.

One thing this trades away: Gym's selection is *sticky* round-robin -- a session keeps its
backend -- so per-request least-outstanding gives up prefix-cache affinity across the
turns of a multi-turn rollout. That is a real cost, not purely a defect being fixed, and
it is worth measuring before enabling this on a multi-turn workload.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

import ray

# Hop-by-hop headers are per-connection and must not be forwarded; passing Host through
# would also make the backend see the router's address.
_SKIPPED_REQUEST_HEADERS = frozenset({"host", "content-length", "connection"})
_SKIPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "connection"}
)
# Body chunk size for the streaming pass-through. Large enough that a long completion
# does not cost thousands of iterations, small enough not to buffer a whole response.
_STREAM_CHUNK_BYTES = 64 * 1024
_BACKEND_DEADLINE_HEADER = "X-Megatron-Request-Timeout-Seconds"
_ROUTER_TRACE_HEADER = "X-Nemo-Router-Request-Id"
_RETRYABLE_HEADER = "X-Nemo-Retryable"
_ERROR_CODE_HEADER = "X-Nemo-Error-Code"
_GENERATION_ABORTED_ERROR_CODE = "generation_aborted"
_REQUEST_BODY_TIMEOUT_ERROR_CODE = "request_body_timeout"
_ROUTER_ADMISSION_TIMEOUT_ERROR_CODE = "router_admission_timeout"
_ADMISSION_BACKEND = "<waiting>"


def _alphabetic_tag(value: int) -> str:
    """Return a compact letters-only counter that Ray log dedup cannot normalize."""
    value = int(value)
    chars = []
    while True:
        value, remainder = divmod(value, 26)
        chars.append(chr(ord("a") + remainder))
        if value == 0:
            return "".join(reversed(chars))
        value -= 1


class _AdmissionLease:
    """One router admission, released exactly once across cancellation paths."""

    def __init__(
        self,
        manager: "_RouterAdmission",
        backend: str,
        reserved_bytes: int,
    ) -> None:
        self._manager = manager
        self.backend = backend
        self.reserved_bytes = reserved_bytes
        self._bytes_released = False
        self._request_released = False

    async def grow_bytes(self, additional_bytes: int) -> None:
        """Reserve bytes not covered by an unknown-length request's estimate."""
        if additional_bytes <= 0 or self._bytes_released:
            return
        await self._manager.grow_bytes(additional_bytes)
        self.reserved_bytes += additional_bytes

    async def release_body_bytes(self) -> None:
        """Release the byte budget once request-body forwarding reaches EOF."""
        if self._bytes_released:
            return
        self._bytes_released = True
        await self._manager.release_bytes(self.reserved_bytes)

    async def release(self) -> None:
        """Release request/backend capacity and any remaining byte reservation."""
        if self._request_released:
            return
        self._request_released = True
        bytes_to_release = 0 if self._bytes_released else self.reserved_bytes
        self._bytes_released = True
        await self._manager.release_request(self.backend, bytes_to_release)


class _RouterAdmission:
    """Atomic count- and byte-weighted admission for router forwarding."""

    def __init__(
        self,
        *,
        backends: list[str],
        max_requests: int,
        max_requests_per_backend: int,
        max_bytes: int,
    ) -> None:
        self._max_requests = max_requests
        self._max_requests_per_backend = max_requests_per_backend
        self._max_bytes = max_bytes
        self._requests = 0
        self._bytes = 0
        self._waiters = 0
        self._requests_by_backend = {backend: 0 for backend in backends}
        self._condition = asyncio.Condition()

    async def acquire(
        self,
        serving_backends: Callable[[], list[str]],
        reserved_bytes: int,
    ) -> _AdmissionLease:
        """Wait for all limits, then atomically reserve a least-loaded backend."""
        async with self._condition:
            self._waiters += 1
            try:
                while True:
                    eligible = [
                        backend
                        for backend in serving_backends()
                        if self._requests_by_backend[backend]
                        < self._max_requests_per_backend
                    ]
                    if (
                        eligible
                        and self._requests < self._max_requests
                        and self._bytes + reserved_bytes <= self._max_bytes
                    ):
                        backend = min(
                            eligible,
                            key=lambda url: (self._requests_by_backend[url], url),
                        )
                        self._requests += 1
                        self._bytes += reserved_bytes
                        self._requests_by_backend[backend] += 1
                        return _AdmissionLease(self, backend, reserved_bytes)

                    # Membership updates arrive on the Ray actor thread and cannot
                    # notify this event-loop condition directly. A bounded wake makes
                    # a newly healthy backend visible without cross-thread operations.
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=1.0)
                    except TimeoutError:
                        pass
            finally:
                self._waiters -= 1

    async def grow_bytes(self, additional_bytes: int) -> None:
        """Extend an unknown-length request's reservation without oversubscription."""
        async with self._condition:
            while self._bytes + additional_bytes > self._max_bytes:
                await self._condition.wait()
            self._bytes += additional_bytes

    async def release_bytes(self, reserved_bytes: int) -> None:
        async with self._condition:
            self._bytes = max(0, self._bytes - reserved_bytes)
            self._condition.notify_all()

    async def release_request(self, backend: str, reserved_bytes: int) -> None:
        async with self._condition:
            self._requests = max(0, self._requests - 1)
            self._bytes = max(0, self._bytes - reserved_bytes)
            self._requests_by_backend[backend] = max(
                0, self._requests_by_backend[backend] - 1
            )
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """Return approximate live counters for cross-thread diagnostics."""
        return {
            "enabled": True,
            "requests": self._requests,
            "waiters": self._waiters,
            "max_requests": self._max_requests,
            "bytes": self._bytes,
            "max_bytes": self._max_bytes,
            "requests_by_backend": dict(self._requests_by_backend),
            "max_requests_per_backend": self._max_requests_per_backend,
        }


class GenerationRouterImpl:
    """Routing logic and HTTP server, split out so it is testable without Ray."""

    def __init__(
        self,
        *,
        backend_urls: list[str],
        host: str,
        port: int,
        backend_timeout_s: float,
        connect_timeout_s: float,
        no_healthy_backend_status: int,
        health_managed: bool = False,
        diagnostics_interval_s: Optional[float] = 30.0,
        admission_enabled: bool = False,
        max_inflight_requests: int = 128,
        max_inflight_requests_per_backend: int = 8,
        max_inflight_request_bytes: int = 4 * 1024**3,
        unknown_request_bytes: int = 64 * 1024**2,
        request_body_timeout_s: float = 120.0,
    ) -> None:
        if not backend_urls:
            raise ValueError("GenerationRouter requires at least one backend URL")
        if max_inflight_requests_per_backend > max_inflight_requests:
            raise ValueError(
                "max_inflight_requests_per_backend cannot exceed "
                "max_inflight_requests"
            )
        if unknown_request_bytes > max_inflight_request_bytes:
            raise ValueError(
                "unknown_request_bytes cannot exceed max_inflight_request_bytes"
            )
        self._all_backends = list(backend_urls)
        # Starts as every backend: a restarted router has no health history, and routing
        # to a shard that turns out to be dead is self-correcting on the next push.
        self._serving: list[str] = list(backend_urls)
        self._inflight: dict[str, int] = {url: 0 for url in backend_urls}
        self._backend_failures: dict[str, int] = {url: 0 for url in backend_urls}
        # Counted for the same reason failures are, and drained in the same breath: the
        # reported streak is what condemns a wedged engine, and without a success signal
        # it is monotonic. On the router path nothing else clears it -- report_success has
        # exactly one caller, on the native adapter -- so a healthy shard collecting
        # unrelated blips days apart walks 1, 2, 3 and is condemned with thousands of
        # successes in between.
        self._backend_successes: dict[str, int] = {url: 0 for url in backend_urls}
        self._host = host
        self._port = port
        self._backend_timeout_s = backend_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._no_healthy_backend_status = no_healthy_backend_status
        # Whether a GenerationFleetHealth is driving set_serving_backends. It gates the
        # reflex drop in _handle: dropping a backend locally is only safe because a later
        # membership push puts it back. With no monitor nothing ever pushes, so the drop
        # would be permanent and a few transient blips would drain the fleet to nothing.
        self._health_managed = health_managed
        self._requests_total = 0
        self._no_backend_total = 0
        self._backend_error_total = 0
        self._responses_total = 0
        self._last_request_at: Optional[float] = None
        self._last_response_at: Optional[float] = None
        self._next_flow_request_id = 0
        # backend, path, request start, current phase, phase start. Keeping the phase
        # here distinguishes requests that have not received backend headers (frontend
        # queue/admission/generation) from replies whose bodies are stuck in transport.
        self._active_requests: dict[int, tuple[str, str, float, str, float]] = {}
        # The aiohttp loop runs on a daemon thread while Ray invokes metrics() on the
        # actor thread. Keep snapshots coherent instead of iterating a mutating dict.
        self._flow_lock = threading.Lock()
        self._diagnostics_interval_s = diagnostics_interval_s
        self._unknown_request_bytes = unknown_request_bytes
        self._request_body_timeout_s = request_body_timeout_s
        self._max_inflight_requests = max_inflight_requests
        self._max_inflight_requests_per_backend = (
            max_inflight_requests_per_backend
        )
        self._admission = (
            _RouterAdmission(
                backends=backend_urls,
                max_requests=max_inflight_requests,
                max_requests_per_backend=max_inflight_requests_per_backend,
                max_bytes=max_inflight_request_bytes,
            )
            if admission_enabled
            else None
        )
        self._thread: Optional[threading.Thread] = None
        self._socket: Any = None

    def base_url(self) -> str:
        """The single URL handed to NeMo-Gym. Stable for the life of the run.

        A method rather than a property so the Ray actor can expose it remotely.
        """
        return f"http://{self._host}:{self._port}/v1"

    def set_serving_backends(self, urls: list[str]) -> None:
        """Replace the eligible backend set.

        Takes the full set rather than a delta, so a missed update, a reordered one, or a
        restarted router all converge on the next push instead of needing sequence
        numbers and replay.
        """
        eligible = [url for url in urls if url in self._inflight]
        unknown = [url for url in urls if url not in self._inflight]
        if unknown:
            # A URL-normalisation divergence between the ports reserved before load and
            # the URLs the monitor reports after it would otherwise show up only as
            # permanent 409s, with nothing anywhere saying why.
            print(
                f"policy router: ignoring {len(unknown)} pushed URL(s) it does not "
                f"serve: {unknown}; known backends: {self._all_backends}",
                flush=True,
            )
        # Rebound rather than mutated: the server thread reads this reference without a
        # lock, and swapping it wholesale means a reader always sees a consistent list.
        self._serving = eligible

    def drain_backend_outcomes(self) -> dict[str, tuple[int, int]]:
        """Hand over per-backend ``(successes, failures)`` since the last drain, and reset.

        The router sees failures no liveness probe can -- a wedged engine answers
        ``is_alive`` from a healthy worker process. It holds no monitor reference by
        design, so instead of reporting, it counts, and the controller's probe tick drains
        these into the fleet ledger.

        Successes travel with them rather than in a separate call, because the ledger needs
        both halves of the same window to decide anything: a failure count alone cannot
        tell a shard that failed three times running from one that failed three times among
        thousands of successes. Draining them apart would let those windows interleave and
        reintroduce exactly the bug this fixes.

        Only backends with something to report appear, so a quiet window drains empty.
        """
        urls = {
            url
            for url, n in (
                *self._backend_successes.items(),
                *self._backend_failures.items(),
            )
            if n
        }
        outcomes = {
            url: (
                self._backend_successes.get(url, 0),
                self._backend_failures.get(url, 0),
            )
            for url in urls
        }
        for url in urls:
            self._backend_successes[url] = 0
            self._backend_failures[url] = 0
        return outcomes

    def metrics(self) -> dict[str, float]:
        snapshot = self.diagnostics()
        return {
            "router/requests_total": float(snapshot["requests_total"]),
            "router/responses_total": float(snapshot["responses_total"]),
            "router/no_healthy_backend_total": float(snapshot["no_backend_total"]),
            "router/backend_error_total": float(snapshot["backend_error_total"]),
            "router/serving_backends": float(snapshot["serving_backends"]),
            "router/inflight_requests": float(snapshot["inflight_requests"]),
            "router/oldest_inflight_seconds": float(
                snapshot["oldest_inflight_seconds"]
            ),
            "router/seconds_since_last_request": float(
                snapshot["seconds_since_last_request"]
            ),
            "router/seconds_since_last_response": float(
                snapshot["seconds_since_last_response"]
            ),
            "router/admitted_requests": float(snapshot["admission"]["requests"]),
            "router/admission_waiters": float(snapshot["admission"]["waiters"]),
            "router/admitted_request_bytes": float(snapshot["admission"]["bytes"]),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return a coherent request-flow snapshot for watchdogs and heartbeats."""
        now = time.monotonic()
        with self._flow_lock:
            active = list(self._active_requests.values())
            active_by_backend = {
                backend: sum(
                    1
                    for active_backend, _, _, _, _ in active
                    if active_backend == backend
                )
                for backend in self._all_backends
            }
            oldest_by_backend = {
                backend: max(
                    (
                        now - started
                        for active_backend, _, started, _, _ in active
                        if active_backend == backend
                    ),
                    default=0.0,
                )
                for backend in self._all_backends
            }
            active_by_phase: dict[str, int] = {}
            for _, _, _, phase, _ in active:
                active_by_phase[phase] = active_by_phase.get(phase, 0) + 1
            oldest_requests = sorted(
                (
                    {
                        "backend": backend,
                        "path": path,
                        "age_seconds": now - started,
                        "phase": phase,
                        "phase_age_seconds": now - phase_started,
                    }
                    for backend, path, started, phase, phase_started in active
                ),
                key=lambda item: item["age_seconds"],
                reverse=True,
            )[:5]
            snapshot = {
                "requests_total": self._requests_total,
                "responses_total": self._responses_total,
                "no_backend_total": self._no_backend_total,
                "backend_error_total": self._backend_error_total,
                "serving_backends": len(self._serving),
                "inflight_requests": len(active),
                "oldest_inflight_seconds": (
                    oldest_requests[0]["age_seconds"] if oldest_requests else 0.0
                ),
                "seconds_since_last_request": (
                    now - self._last_request_at
                    if self._last_request_at is not None
                    else -1.0
                ),
                "seconds_since_last_response": (
                    now - self._last_response_at
                    if self._last_response_at is not None
                    else -1.0
                ),
                "active_by_backend": active_by_backend,
                "oldest_by_backend": oldest_by_backend,
                "active_by_phase": active_by_phase,
                "oldest_requests": oldest_requests,
            }
        snapshot["admission"] = (
            self._admission.snapshot()
            if self._admission is not None
            else {
                "enabled": False,
                "requests": 0,
                "waiters": 0,
                "max_requests": 0,
                "bytes": 0,
                "max_bytes": 0,
                "requests_by_backend": {},
                "max_requests_per_backend": 0,
            }
        )
        return snapshot

    def _set_request_phase(self, flow_request_id: int, phase: str) -> None:
        """Advance one router request's lifecycle phase for live diagnostics."""
        with self._flow_lock:
            active = self._active_requests.get(flow_request_id)
            if active is None:
                return
            backend, path, started, _, _ = active
            self._active_requests[flow_request_id] = (
                backend,
                path,
                started,
                phase,
                time.monotonic(),
            )

    def _set_request_backend(self, flow_request_id: int, backend: str) -> None:
        """Assign the backend chosen atomically by the admission controller."""
        with self._flow_lock:
            active = self._active_requests.get(flow_request_id)
            if active is None:
                return
            _, path, started, phase, phase_started = active
            self._active_requests[flow_request_id] = (
                backend,
                path,
                started,
                phase,
                phase_started,
            )

    def _request_phase(self, flow_request_id: int) -> str:
        with self._flow_lock:
            active = self._active_requests.get(flow_request_id)
            return active[3] if active is not None else "unknown"

    def _pick_backend(self) -> Optional[str]:
        """Least-outstanding among eligible backends, or None if there are none."""
        serving = self._serving
        if not serving:
            return None
        return min(serving, key=lambda url: (self._inflight.get(url, 0), url))

    @staticmethod
    def _target_url(backend: str, path_qs: str) -> str:
        """Map an inbound path onto a backend.

        Backends are advertised as ``http://host:port/v1`` while inbound paths already
        carry their own prefix -- ``/v1/chat/completions`` for most calls, but bare
        ``/tokenize`` because Gym's ``create_tokenize`` strips ``/v1`` first. Stripping
        the suffix and appending the full path handles both.
        """
        return backend.removesuffix("/v1") + path_qs

    async def _handle(self, request: Any) -> Any:
        from aiohttp import ClientError, web

        now = time.monotonic()
        with self._flow_lock:
            self._requests_total += 1
            self._last_request_at = now
            flow_request_id = self._next_flow_request_id
            self._next_flow_request_id += 1
            self._active_requests[flow_request_id] = (
                _ADMISSION_BACKEND,
                request.rel_url.path,
                now,
                (
                    "waiting_for_router_admission"
                    if self._admission is not None
                    else "forward_start"
                ),
                now,
            )
        if not self._serving:
            with self._flow_lock:
                self._active_requests.pop(flow_request_id, None)
                self._no_backend_total += 1
                self._responses_total += 1
                self._last_response_at = time.monotonic()
            # The status matters: NeMo-Gym retries 429/500/502/503/504/520, and for the
            # rate-limit codes it *raises its own retry ceiling* each time, so returning
            # one of those would spin forever. This code must stay outside that set.
            return web.json_response(
                {
                    "error": "no healthy generation backend",
                    "backends": self._all_backends,
                },
                status=self._no_healthy_backend_status,
            )

        content_length = request.content_length
        reserved_bytes = (
            content_length if content_length is not None else self._unknown_request_bytes
        )
        admission_snapshot = (
            self._admission.snapshot() if self._admission is not None else None
        )
        if (
            admission_snapshot is not None
            and reserved_bytes > admission_snapshot["max_bytes"]
        ):
            with self._flow_lock:
                self._active_requests.pop(flow_request_id, None)
                self._responses_total += 1
                self._last_response_at = time.monotonic()
            return web.json_response(
                {
                    "error": {
                        "message": (
                            f"request body reservation {reserved_bytes} exceeds router "
                            f"byte budget {admission_snapshot['max_bytes']}"
                        ),
                        "type": "RequestEntityTooLarge",
                        "retryable": False,
                    }
                },
                status=413,
                headers={_RETRYABLE_HEADER: "false"},
            )

        lease: Optional[_AdmissionLease] = None
        backend = _ADMISSION_BACKEND
        outcome = "cancelled"
        try:
            if self._admission is not None:
                async with asyncio.timeout(self._request_body_timeout_s):
                    lease = await self._admission.acquire(
                        lambda: self._serving, reserved_bytes
                    )
                backend = lease.backend
            else:
                selected_backend = self._pick_backend()
                if selected_backend is None:
                    raise RuntimeError("serving backend set became empty during routing")
                backend = selected_backend
            self._set_request_backend(flow_request_id, backend)
            self._set_request_phase(flow_request_id, "forward_start")
            with self._flow_lock:
                self._inflight[backend] = self._inflight.get(backend, 0) + 1
            response = await self._forward(
                request, backend, flow_request_id, now, lease
            )
            outcome = f"http-{response.status}"
            return response
        except (TimeoutError, ClientError) as error:
            outcome = type(error).__name__
            return self._on_backend_error(
                backend,
                error,
                elapsed_seconds=time.monotonic() - now,
                request_phase=self._request_phase(flow_request_id),
            )
        finally:
            if lease is not None:
                await lease.release()
            finished_at = time.monotonic()
            elapsed = finished_at - now
            with self._flow_lock:
                self._active_requests.pop(flow_request_id, None)
                if backend != _ADMISSION_BACKEND:
                    self._inflight[backend] = max(
                        0, self._inflight.get(backend, 0) - 1
                    )
                self._responses_total += 1
                self._last_response_at = finished_at
            if elapsed >= 60.0 or outcome != "http-200":
                backend_name = (
                    f"b{self._all_backends.index(backend)}"
                    if backend in self._all_backends
                    else backend
                )
                print(
                    "policy router request done: "
                    f"tag={_alphabetic_tag(flow_request_id)} id={flow_request_id} "
                    f"backend={backend_name} path={request.rel_url.path} "
                    f"elapsed={elapsed:.1f}s outcome={outcome}",
                    flush=True,
                )

    def _on_backend_error(
        self,
        backend: str,
        error: BaseException,
        *,
        elapsed_seconds: float,
        request_phase: str = "unknown",
    ) -> Any:
        """Answer for a backend that failed, deliberately rather than by accident.

        Without this, aiohttp answers instead, and its choice of status decides whether
        the run survives. A wedged backend trips the client timeout, which aiohttp
        reports as **504** -- and 504 is in NeMo-Gym's rate-limit retry subset, where
        ``_request_with_retry`` raises its own ceiling on every attempt. That is an
        unbounded retry loop at ``backend_timeout_s`` per turn: exactly the hang
        ``_check_status_is_not_retried_by_gym`` exists to prevent, reintroduced through
        the error path that validator never covered.

        500 instead, because it is in Gym's *bounded* retry set: Gym re-sends this one
        HTTP call, the next _pick_backend lands on a healthy shard, and a multi-turn
        rollout keeps the turns it had already completed. The no-healthy-backend status
        (409) would be wrong here -- not retried, so it fails the whole rollout and every
        turn is redone from scratch by the row re-dispatch a layer up.
        """
        from aiohttp import web

        with self._flow_lock:
            self._backend_error_total += 1
        pre_submit_failure = request_phase in {
            "waiting_for_router_admission",
            "streaming_request_body",
        }
        if not pre_submit_failure:
            self._backend_failures[backend] = (
                self._backend_failures.get(backend, 0) + 1
            )
        # The transport cause, logged here because nothing else keeps it. It goes into
        # the response body, which is Gym's to interpret, and the ledger only ever sees
        # the aggregated "N failed request(s)" summary -- so without this line a
        # condemned shard's record cannot say whether it refused connections, reset them,
        # or timed out, which are three different problems.
        if isinstance(error, TimeoutError):
            print(
                "POLICY ROUTER BACKEND TIMEOUT: "
                f"backend={backend} phase={request_phase} "
                f"elapsed_s={elapsed_seconds:.3f} "
                f"configured_timeout_s={self._backend_timeout_s:.3f} "
                "classification=local_deadline_exceeded "
                "backend_exception_observed=false",
                flush=True,
            )
        else:
            print(
                "POLICY ROUTER BACKEND TRANSPORT ERROR: "
                f"backend={backend} phase={request_phase} "
                f"elapsed_s={elapsed_seconds:.3f} "
                f"error_type={type(error).__name__} error={error!r}",
                flush=True,
            )
        if self._health_managed and not pre_submit_failure:
            # Reflex: stop routing here until the next membership push re-adds it.
            # Rebound, not mutated -- same reason as set_serving_backends, and this runs
            # on the server thread while pushes arrive on the actor's.
            self._serving = [url for url in self._serving if url != backend]
        status = 500 if self._serving else self._no_healthy_backend_status
        error_payload: dict[str, Any] = {
            "message": f"router/backend failed: {type(error).__name__}: {error}",
            "type": type(error).__name__,
            "retryable": True,
        }
        response_headers = None
        is_connect_failure = type(error).__name__ in {
            "ClientConnectorError",
            "ConnectionTimeoutError",
        } or (
            isinstance(error, TimeoutError)
            and request_phase == "forward_start"
            and elapsed_seconds <= self._connect_timeout_s + 1.0
        )
        if isinstance(error, TimeoutError) and not is_connect_failure:
            if pre_submit_failure:
                # The frontend cannot parse or admit incomplete JSON, and an admission
                # waiter has not opened a backend request at all. Both are safe to retry
                # after bounded backoff and must not quarantine a backend.
                error_code = (
                    _ROUTER_ADMISSION_TIMEOUT_ERROR_CODE
                    if request_phase == "waiting_for_router_admission"
                    else _REQUEST_BODY_TIMEOUT_ERROR_CODE
                )
                error_payload.update(
                    code=error_code,
                    retryable=True,
                )
                response_headers = {
                    _RETRYABLE_HEADER: "true",
                    _ERROR_CODE_HEADER: error_code,
                }
            else:
                # A timed-out generation may already have been admitted. Retrying it
                # here creates duplicate engine work; the frontend deadline normally
                # aborts it first. Structured transport metadata is the fallback
                # against retry amplification if that response misses this proxy.
                error_payload.update(
                    code=_GENERATION_ABORTED_ERROR_CODE,
                    retryable=False,
                )
                response_headers = {
                    _RETRYABLE_HEADER: "false",
                    _ERROR_CODE_HEADER: _GENERATION_ABORTED_ERROR_CODE,
                }
        return web.json_response(
            {
                "error": error_payload,
                "backend": backend,
            },
            status=status,
            headers=response_headers,
        )

    async def _forward(
        self,
        request: Any,
        backend: str,
        flow_request_id: int,
        started_at: float,
        admission_lease: Optional[_AdmissionLease],
    ) -> Any:
        from aiohttp import ClientTimeout, web

        session = request.app["session"]
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _SKIPPED_REQUEST_HEADERS
        }
        # Expire inside the inference frontend first, while this connection is
        # alive, so it can send ABORT_REQUEST to the coordinator. Closing only
        # this proxy hop leaves an orphan generation consuming engine capacity.
        frontend_budget_s = max(
            1.0,
            self._backend_timeout_s - (time.monotonic() - started_at) - 5.0,
        )
        headers[_BACKEND_DEADLINE_HEADER] = f"{frontend_budget_s:.3f}"
        headers[_ROUTER_TRACE_HEADER] = str(flow_request_id)

        request_body_bytes = 0

        async def _request_body():
            nonlocal request_body_bytes
            self._set_request_phase(flow_request_id, "streaming_request_body")
            async with asyncio.timeout(self._request_body_timeout_s):
                async for chunk in request.content.iter_chunked(_STREAM_CHUNK_BYTES):
                    request_body_bytes += len(chunk)
                    if admission_lease is not None:
                        unreserved_bytes = (
                            request_body_bytes - admission_lease.reserved_bytes
                        )
                        if unreserved_bytes > 0:
                            await admission_lease.grow_bytes(unreserved_bytes)
                    yield chunk
            if admission_lease is not None:
                await admission_lease.release_body_bytes()
            self._set_request_phase(flow_request_id, "awaiting_headers")
            print(
                "policy router request body sent: "
                f"tag={_alphabetic_tag(flow_request_id)} id={flow_request_id} "
                f"backend={backend} bytes={request_body_bytes} "
                f"elapsed={time.monotonic() - started_at:.3f}s",
                flush=True,
            )

        async with session.request(
            method=request.method,
            url=self._target_url(backend, request.rel_url.path_qs),
            headers=headers,
            data=_request_body(),
            # The timeout Gym's own client never sets. Without it a wedged backend holds
            # this request, and the rollout behind it, indefinitely.
            #
            # total must cover the whole generation: Gym pins stream=false, so no bytes
            # arrive until the completion finishes and an idle-read timeout would kill
            # long generations -- elapsed-total is the only wedge detector this hop can
            # have. The handshake is the opposite: a connect to a local vLLM either
            # completes in milliseconds or never will, so giving it the full budget just
            # means a black-holed SYN (node gone, no RST) parks the rollout for the
            # whole 600s.
            timeout=ClientTimeout(
                total=self._backend_timeout_s, sock_connect=self._connect_timeout_s
            ),
        ) as upstream:
            headers_at = time.monotonic()
            self._set_request_phase(flow_request_id, "streaming_body")
            print(
                "policy router upstream headers: "
                f"tag={_alphabetic_tag(flow_request_id)} id={flow_request_id} "
                f"backend={backend} status={upstream.status} "
                f"request_bytes={request_body_bytes} "
                f"headers_after={headers_at - started_at:.3f}s",
                flush=True,
            )
            if upstream.status >= 500:
                print(
                    "POLICY ROUTER BACKEND HTTP ERROR RESPONSE: "
                    f"tag={_alphabetic_tag(flow_request_id)} id={flow_request_id} "
                    f"backend={backend} status={upstream.status} "
                    f"elapsed_s={headers_at - started_at:.3f} "
                    f"error_code={upstream.headers.get(_ERROR_CODE_HEADER, 'unspecified')} "
                    f"retryable={upstream.headers.get(_RETRYABLE_HEADER, 'unspecified')}",
                    flush=True,
                )
            response = web.StreamResponse(
                status=upstream.status,
                headers={
                    key: value
                    for key, value in upstream.headers.items()
                    if key.lower() not in _SKIPPED_RESPONSE_HEADERS
                },
            )
            prepare_started_at = time.monotonic()
            await response.prepare(request)
            prepared_at = time.monotonic()
            self._set_request_phase(flow_request_id, "relaying_response_body")
            print(
                "policy router downstream headers sent: "
                f"tag={_alphabetic_tag(flow_request_id)} id={flow_request_id} "
                f"backend={backend} prepare_ms={(prepared_at - prepare_started_at) * 1000:.1f} "
                f"elapsed={prepared_at - started_at:.3f}s",
                flush=True,
            )
            # Streamed rather than buffered: a completion carrying per-token logprobs is
            # large, and this sits on every rollout's critical path.
            response_body_bytes = 0
            response_chunks = 0
            async for chunk in upstream.content.iter_chunked(_STREAM_CHUNK_BYTES):
                response_body_bytes += len(chunk)
                response_chunks += 1
                await response.write(chunk)
            self._set_request_phase(flow_request_id, "write_eof")
            eof_started_at = time.monotonic()
            await response.write_eof()
            finished_at = time.monotonic()
            print(
                "policy router response relayed: "
                f"tag={_alphabetic_tag(flow_request_id)} id={flow_request_id} "
                f"backend={backend} chunks={response_chunks} bytes={response_body_bytes} "
                f"body_ms={(eof_started_at - prepared_at) * 1000:.1f} "
                f"eof_ms={(finished_at - eof_started_at) * 1000:.1f} "
                f"total_ms={(finished_at - started_at) * 1000:.1f}",
                flush=True,
            )
            # After write_eof, so it means "served a whole response" rather than "accepted
            # the request".
            self._backend_successes[backend] = (
                self._backend_successes.get(backend, 0) + 1
            )
            return response

    def build_app(self) -> Any:
        """Build the aiohttp application serving Gym's endpoint surface."""
        from aiohttp import ClientSession, TCPConnector, web

        app = web.Application()

        async def _open_session(app_: Any) -> None:
            # Explicit admission above this connector owns normal queueing and exposes
            # its phase. Matching connector limits are a defensive invariant: a future
            # call site cannot silently bypass admission and recreate an unbounded
            # multi-gigabyte upload fan-out.
            app_["session"] = ClientSession(
                connector=TCPConnector(
                    limit=(
                        self._max_inflight_requests
                        if self._admission is not None
                        else 0
                    ),
                    limit_per_host=(
                        self._max_inflight_requests_per_backend
                        if self._admission is not None
                        else 0
                    ),
                )
            )

        async def _close_session(app_: Any) -> None:
            await app_["session"].close()

        app.on_startup.append(_open_session)
        app.on_cleanup.append(_close_session)

        # Exactly the calls NeMo-Gym's NeMoGymAsyncOpenAI makes. /tokenize is not under
        # /v1 because create_tokenize strips the suffix before appending.
        #
        # Deliberately an allowlist: this router's URL becomes Gym's *global*
        # policy_base_url, and some Gym envs point other surfaces at it -- speed_bench
        # scrapes GET /metrics, the claude-code agent POSTs /v1/messages. Those get 404
        # here where a raw vLLM URL answered, so run those envs with the router off.
        # Forwarding /metrics would be worse than refusing it: each shard keeps its own
        # counters, so a routed scrape returns one arbitrary shard's numbers as though
        # they were the fleet's.
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/models"):
            app.router.add_route("*", path, self._handle)
        app.router.add_route("*", "/tokenize", self._handle)
        return app

    async def _diagnostic_heartbeat(self) -> None:
        """Print centralized request flow even when every downstream log is quiet."""
        assert self._diagnostics_interval_s is not None
        previous_requests = 0
        previous_responses = 0
        heartbeat_index = 0
        while True:
            await asyncio.sleep(self._diagnostics_interval_s)
            snapshot = self.diagnostics()
            requests = int(snapshot["requests_total"])
            responses = int(snapshot["responses_total"])
            active = [
                (
                    index,
                    int(snapshot["active_by_backend"][backend]),
                    float(snapshot["oldest_by_backend"][backend]),
                )
                for index, backend in enumerate(self._all_backends)
                if snapshot["active_by_backend"][backend]
            ]
            active_text = (
                ",".join(
                    f"b{index}:{count}@{oldest:.0f}s" for index, count, oldest in active
                )
                or "none"
            )
            print(
                "policy router heartbeat: "
                f"tag={_alphabetic_tag(heartbeat_index)} "
                f"requests={requests} (+{requests - previous_requests}) "
                f"responses={responses} (+{responses - previous_responses}) "
                f"inflight={snapshot['inflight_requests']} "
                f"oldest={snapshot['oldest_inflight_seconds']:.0f}s "
                f"request_idle={snapshot['seconds_since_last_request']:.0f}s "
                f"response_idle={snapshot['seconds_since_last_response']:.0f}s "
                f"serving={snapshot['serving_backends']}/{len(self._all_backends)} "
                f"admitted={snapshot['admission']['requests']}/"
                f"{snapshot['admission']['max_requests']} "
                f"admission_waiters={snapshot['admission']['waiters']} "
                f"admitted_bytes={snapshot['admission']['bytes']}/"
                f"{snapshot['admission']['max_bytes']} "
                f"admitted_by_backend="
                f"{snapshot['admission']['requests_by_backend']} "
                f"phases={snapshot['active_by_phase']} "
                f"active=[{active_text}] "
                f"oldest_requests={snapshot['oldest_requests']}",
                flush=True,
            )
            heartbeat_index += 1
            previous_requests = requests
            previous_responses = responses

    def serve_in_background(self) -> None:
        """Run the HTTP server on a daemon thread with its own event loop.

        The socket is bound **here**, synchronously, before the thread starts. Bound
        inside the thread instead, a port conflict raises on a daemon thread nobody
        awaits: the actor stays alive, ``base_url()`` is a pure string format so it keeps
        resolving, and setup's "fail here rather than inside Gym" guard never notices.
        Gym is then handed a URL with no listener and retries the refused connection in
        an uncapped loop -- the exact wedge this router exists to prevent. Binding first
        turns that into a failed actor construction with the port in the traceback.

        Same shape as the vLLM workers handing their reserved socket to uvicorn. Restart
        stays correct: the replacement process rebinds the port its dead predecessor
        freed.
        """
        import socket

        from aiohttp import web

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(128)
        self._socket = sock

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            runner = web.AppRunner(self.build_app(), access_log=None)
            loop.run_until_complete(runner.setup())
            site = web.SockSite(runner, sock)
            loop.run_until_complete(site.start())
            print(f"policy router listening on {self.base_url()}", flush=True)
            if self._diagnostics_interval_s is not None:
                loop.create_task(self._diagnostic_heartbeat())
            loop.run_forever()

        self._thread = threading.Thread(target=_run, name="policy-router", daemon=True)
        self._thread.start()

    def is_serving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=-1)  # pragma: no cover
class GenerationRouterActor(GenerationRouterImpl):
    """Ray actor wrapper. Everything it needs is built in ``__init__``.

    ``max_restarts=-1`` is only meaningful because of that: Ray recreates a restarted
    actor through ``__init__`` alone, so a class that starts its server from a separate
    method comes back permanently broken.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.serve_in_background()
