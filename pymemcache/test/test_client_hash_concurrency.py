"""Concurrency tests for HashClient node health coordination.

These tests cover the coordinated failure/recovery state machine:

* only one caller at a time may run the recovery probe against a node whose
  retry window has elapsed;
* a failed probe immediately re-enters the failed state and re-arms the retry
  window;
* a success from an earlier request never clears a newer failure state;
* single-key, batch and multi-node operations all share the same gate, while
  healthy nodes are not serialized and no network call happens while the
  state lock is held.
"""

import socket
import threading
import time
from collections import Counter

import pytest

from pymemcache.client.hash import HashClient
from pymemcache.exceptions import MemcacheUnknownError


SERVER = ("127.0.0.1", 11211)


class _MockClient:
    """Minimal stand-in for a node client with a controllable ``get``."""

    def __init__(self, server=SERVER):
        self.server = server
        self.calls = 0
        self.side_effect = None

    def _invoke(self, *args, **kwargs):
        self.calls += 1
        if self.side_effect is None:
            return "value"
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        return self.side_effect

    def get(self, *args, **kwargs):
        return self._invoke(*args, **kwargs)

    def set(self, *args, **kwargs):
        self._invoke(*args, **kwargs)
        return True

    def set_many(self, values, *args, **kwargs):
        self._invoke(values, *args, **kwargs)
        return []

    def get_many(self, keys, *args, **kwargs):
        self._invoke(keys, *args, **kwargs)
        return {key: "value" for key in keys}

    def close(self):
        pass


def _make_client(mock_client=None, **kwargs):
    kwargs.setdefault("retry_attempts", 3)
    # A retry window of zero makes every gated failure immediately eligible
    # for a probe unless one is already in progress.
    kwargs.setdefault("retry_timeout", 0)
    client = HashClient([], **kwargs)
    mock_client = mock_client or _MockClient()
    key = client._make_client_key(SERVER)
    client.clients[key] = mock_client
    client.hasher.add_node(key)
    return client, mock_client


def _mark_failed(client, server=SERVER):
    """Put a node in the failed state with an elapsed retry window."""
    client._mark_failed_server(server)
    metadata = client._failed_clients[server]
    # Ensure the retry window has elapsed so a probe is permitted.
    metadata["failed_time"] = 0.0
    return metadata


@pytest.mark.unit()
class TestHashClientConcurrency:
    def test_only_single_probe_after_retry_timeout(self):
        client, mock = _make_client(ignore_exc=True)
        _mark_failed(client)

        nthreads = 8
        start = threading.Barrier(nthreads)
        probe_entered = threading.Event()
        release_probe = threading.Event()
        results = []

        def side_effect(*args, **kwargs):
            probe_entered.set()
            assert release_probe.wait(timeout=5)
            return "probed"

        mock.side_effect = side_effect

        def worker():
            start.wait()
            results.append(client.get("key"))

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for thread in threads:
            thread.start()

        # Exactly one caller becomes the probe and enters the network call.
        assert probe_entered.wait(timeout=5)

        # While the probe is blocked, the other callers must observe the
        # in-progress probe, skip the node and return without contacting it.
        time.sleep(0.5)
        assert mock.calls == 1
        assert all(result is None for result in results)

        release_probe.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        assert mock.calls == 1
        # The probe recovered the node, every caller got a coherent result.
        assert SERVER not in client._failed_clients
        assert Counter(results) == Counter(
            ["probed"] + [None] * (nthreads - 1)
        )

    def test_probe_in_progress_follows_ignore_exc_defaults(self):
        # ignore_exc=True: waiters return the command's default value.
        client, mock = _make_client(ignore_exc=True)
        metadata = _mark_failed(client)

        metadata["probing"] = True
        assert client.get("key", default="fallback") == "fallback"
        assert client.set("key", "value") is False
        assert client.get_many(["key"]) == {}
        assert client.set_many({"key": "value"}) == ["key"]
        assert mock.calls == 0

    def test_probe_in_progress_skips_regardless_of_ignore_exc(self):
        # Historical semantics: a node inside its retry window (or already
        # being probed) is simply skipped and the default is returned
        # regardless of ignore_exc; ignore_exc only governs raised
        # exceptions and the out-of-servers case.
        client, mock = _make_client(ignore_exc=False)
        metadata = _mark_failed(client)

        metadata["probing"] = True
        assert client.get("key") is None
        assert client.set("key", "value") is False
        assert mock.calls == 0

    def test_failed_probe_re_enters_failed_state(self):
        client, mock = _make_client(ignore_exc=False, retry_timeout=60)
        metadata = _mark_failed(client)
        generation = metadata["generation"]
        mock.side_effect = socket.timeout()

        with pytest.raises(socket.timeout):
            client.get("key")

        metadata = client._failed_clients[SERVER]
        assert metadata["probing"] is False
        assert metadata["attempts"] == 1
        assert metadata["generation"] == generation + 1
        assert SERVER not in client._dead_clients

        # The retry window is re-armed: while it has not elapsed the node is
        # skipped, not contacted.
        calls_before = mock.calls
        assert client.get("key") is None
        assert mock.calls == calls_before

    def test_failed_probe_returns_default_with_ignore_exc(self):
        client, mock = _make_client(ignore_exc=True)
        _mark_failed(client)
        mock.side_effect = OSError("boom")

        assert client.get("key", default="d") == "d"

        metadata = client._failed_clients[SERVER]
        assert metadata["probing"] is False
        assert metadata["attempts"] == 1

    def test_repeated_failed_probes_are_single_flighted(self):
        client, mock = _make_client(ignore_exc=True)
        _mark_failed(client)

        release = threading.Event()
        entered = threading.Event()

        def side_effect(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            raise OSError("still down")

        mock.side_effect = side_effect

        results = []
        start = threading.Barrier(4)

        def worker():
            start.wait()
            results.append(client.get("key"))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()

        assert entered.wait(timeout=5)
        release.set()
        for thread in threads:
            thread.join()

        # One probe attempt despite four concurrent callers.
        assert mock.calls == 1
        assert results == [None] * 4
        metadata = client._failed_clients[SERVER]
        assert metadata["attempts"] == 1
        assert metadata["probing"] is False

    def test_successful_probe_recovers_node(self):
        client, mock = _make_client(ignore_exc=True)
        _mark_failed(client)
        mock.side_effect = "ok"

        assert client.get("key") == "ok"
        assert SERVER not in client._failed_clients

        # Further calls are plain healthy-node calls without gating.
        assert client.get("key") == "ok"
        assert mock.calls == 2

    def test_stale_probe_success_does_not_clear_newer_failure(self):
        client, mock = _make_client(ignore_exc=True)
        metadata = _mark_failed(client)
        stale_generation = metadata["generation"]

        release_stale = threading.Event()
        stale_entered = threading.Event()

        def stale_side_effect(*args, **kwargs):
            stale_entered.set()
            assert release_stale.wait(timeout=5)
            return "stale success"

        mock.side_effect = stale_side_effect

        stale_thread = threading.Thread(target=client.get, args=("key",))
        stale_thread.start()
        assert stale_entered.wait(timeout=5)

        # While the old probe is in flight, the node fails again (a newer
        # failure epoch) via another code path.
        client._mark_failed_server(SERVER)
        newer = client._failed_clients[SERVER]
        assert newer["generation"] == stale_generation + 1
        assert newer["probing"] is False

        # The earlier request now completes successfully; it must not
        # resurrect the node.
        release_stale.set()
        stale_thread.join()

        current = client._failed_clients[SERVER]
        assert current["generation"] == newer["generation"]
        assert client.hasher.get_node("key") is not None

        # The node still fails closed: gated callers skip it until a probe
        # belonging to the new epoch succeeds.
        mock.side_effect = None
        assert client.get("key") == "value"
        assert SERVER not in client._failed_clients

    def test_non_socket_exception_releases_probe_without_state_change(self):
        client, mock = _make_client(ignore_exc=True)
        metadata = _mark_failed(client)
        generation = metadata["generation"]
        mock.side_effect = MemcacheUnknownError("protocol error")

        assert client.get("key") is None

        metadata = client._failed_clients[SERVER]
        assert metadata["probing"] is False
        assert metadata["generation"] == generation
        assert metadata["attempts"] == 0

        # The probe slot is free, so a later call can probe again.
        mock.side_effect = "ok"
        assert client.get("key") == "ok"
        assert SERVER not in client._failed_clients

    def test_non_socket_exception_propagates_without_ignore_exc(self):
        client, mock = _make_client(ignore_exc=False)
        _mark_failed(client)
        mock.side_effect = MemcacheUnknownError("protocol error")

        with pytest.raises(MemcacheUnknownError):
            client.get("key")

        metadata = client._failed_clients[SERVER]
        assert metadata["probing"] is False

    def test_batch_operations_share_the_probe_gate(self):
        client, mock = _make_client(ignore_exc=True)
        _mark_failed(client)

        release = threading.Event()
        entered = threading.Event()

        def blocking_side_effect(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return []

        mock.side_effect = blocking_side_effect

        set_result = []
        get_result = []

        set_thread = threading.Thread(
            target=lambda: set_result.append(
                client.set_many({"a": 1, "b": 2})
            )
        )
        get_thread = threading.Thread(
            target=lambda: get_result.append(client.get_many(["a", "b"]))
        )
        set_thread.start()
        assert entered.wait(timeout=5)
        get_thread.start()

        # The get_many caller observes the set_many probe and skips the node.
        get_thread.join(timeout=5)
        assert not get_thread.is_alive()
        assert get_result == [{}]
        assert mock.calls == 1

        release.set()
        set_thread.join()
        assert mock.calls == 1
        assert set_result == [[]]
        assert SERVER not in client._failed_clients

    def test_healthy_nodes_are_not_serialized_during_probe(self):
        # Traffic against healthy nodes must run concurrently: the per-node
        # coordination state never adds a global serialization point.
        client = HashClient(
            [], retry_attempts=3, retry_timeout=0, ignore_exc=True
        )
        healthy = []
        for port in (11211, 11212):
            node = _MockClient(server=("127.0.0.1", port))
            key = client._make_client_key(node.server)
            client.clients[key] = node
            client.hasher.add_node(key)
            healthy.append(node)

        nthreads = 8
        inside = threading.Barrier(nthreads)
        release = threading.Event()

        def healthy_side_effect(*args, **kwargs):
            inside.wait(timeout=5)
            assert release.wait(timeout=5)
            return "ok"

        for node in healthy:
            node.side_effect = healthy_side_effect

        def worker():
            client.get_many(["a", "b"])

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for thread in threads:
            thread.start()

        # All threads enter the mocked network call concurrently; a serial
        # implementation would make the barrier time out.
        release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()

    def test_network_call_runs_without_state_lock(self):
        client, mock = _make_client(ignore_exc=True)
        _mark_failed(client)

        probe_in_flight = threading.Event()
        release_probe = threading.Event()
        watchdog_result = []

        def watchdog():
            # Wait until the probe is inside its (mocked) network call, then
            # try to take the coordination lock. If the network call held the
            # lock, this non-blocking acquire would fail.
            probe_in_flight.wait(timeout=5)
            watchdog_result.append(
                client._state_lock.acquire(blocking=False)
            )
            if watchdog_result[0]:
                client._state_lock.release()
            release_probe.set()

        def blocking_side_effect(*args, **kwargs):
            probe_in_flight.set()
            assert release_probe.wait(timeout=5)
            return "ok"

        mock.side_effect = blocking_side_effect

        watchdog_thread = threading.Thread(target=watchdog)
        watchdog_thread.start()
        assert client.get("key") == "ok"
        watchdog_thread.join(timeout=5)

        assert watchdog_result == [True]
        assert SERVER not in client._failed_clients

    def test_concurrent_failures_are_recorded_safely(self):
        client, mock = _make_client(ignore_exc=True, retry_timeout=60)
        # Start from an established failure epoch whose retry window has
        # elapsed so exactly one caller wins the probe. A long retry_timeout
        # guarantees that after the probe fails, the freshly re-armed window
        # is still closed, so no other thread can start a second probe: the
        # result is deterministic rather than scheduler-dependent.
        metadata = _mark_failed(client)
        initial_generation = metadata["generation"]

        release = threading.Event()
        entered = threading.Event()

        def failing_side_effect(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            raise OSError("down")

        mock.side_effect = failing_side_effect

        nthreads = 6
        start = threading.Barrier(nthreads)

        def worker():
            start.wait()
            client.get("key")

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for thread in threads:
            thread.start()

        # One caller becomes the probe; the other five observe the in-flight
        # probe and return without touching the node while it is blocked.
        assert entered.wait(timeout=5)
        time.sleep(0.3)
        assert mock.calls == 1

        release.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        metadata = client._failed_clients[SERVER]
        # Single-flighting holds even for failures: exactly one caller became
        # the probe and recorded one new attempt; every other caller skipped
        # the node. State stays coherent with the probe slot released.
        assert metadata["generation"] == initial_generation + 1
        assert metadata["attempts"] == 1
        assert metadata["probing"] is False
        # The node was never moved to the dead pool (retry_attempts=3).
        assert SERVER not in client._dead_clients
        assert mock.calls == 1

    def test_concurrent_dead_reaper_runs_once(self):
        client, mock = _make_client(
            ignore_exc=False, retry_attempts=0, dead_timeout=0
        )
        mock.side_effect = OSError("down")

        with pytest.raises(OSError):
            client.get("key")
        assert SERVER in client._dead_clients
        assert mock.calls == 1

        add_calls = []
        original_add_server = client.add_server

        def counting_add_server(server, *args, **kwargs):
            add_calls.append(server)
            return original_add_server(server, *args, **kwargs)

        client.add_server = counting_add_server

        nthreads = 6
        barrier = threading.Barrier(nthreads)

        def worker():
            barrier.wait()
            client._retry_dead()

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # The locked reaper re-adds the node exactly once.
        assert add_calls == [SERVER]
        assert SERVER not in client._dead_clients
        assert client._make_client_key(SERVER) in client.hasher.nodes

    def test_dead_node_recovery_lifecycle_with_generation(self):
        client, mock = _make_client(
            ignore_exc=True, retry_attempts=1, retry_timeout=60, dead_timeout=0
        )
        mock.side_effect = OSError("down")

        # First failure: failed state, generation 1.
        client.get("key")
        metadata = client._failed_clients[SERVER]
        assert metadata["generation"] == 1
        assert metadata["attempts"] == 0

        # Open the retry window so the next call runs a probe. Its failure
        # records an attempt; the dead transition happens at the following
        # entry-gate check (matching historical behavior).
        metadata["failed_time"] = 0.0
        client.get("key")
        assert client._failed_clients[SERVER]["attempts"] == 1
        assert SERVER not in client._dead_clients

        # Third call: the gate sees the exhausted retry budget and moves the
        # node to the dead pool. That call still runs once and fails again,
        # but no failure metadata is resurrected for a dead node.
        client.get("key")
        assert SERVER in client._dead_clients
        assert SERVER not in client._failed_clients

        # Dead reaper restores the node as a clean healthy node. Reaper's
        # add_server() builds a fresh real client, so re-install the mock to
        # keep driving the node deterministically.
        client._retry_dead()
        client.clients[client._make_client_key(SERVER)] = mock
        assert client._make_client_key(SERVER) in client.hasher.nodes
        assert SERVER not in client._failed_clients

        # A fresh failure starts a brand-new epoch with a full retry budget.
        mock.side_effect = OSError("down")
        client.get("key")
        assert client._failed_clients[SERVER]["generation"] == 1
        assert client._failed_clients[SERVER]["attempts"] == 0

        # A probe belonging to the new epoch recovers the node.
        client._failed_clients[SERVER]["failed_time"] = 0.0
        mock.side_effect = "back"
        assert client.get("key") == "back"
        assert SERVER not in client._failed_clients
