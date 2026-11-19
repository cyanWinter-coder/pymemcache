import collections
import socket
import threading
import time
import logging

from pymemcache.client.base import (
    Client,
    PooledClient,
    check_key_helper,
    normalize_server_spec,
)
from pymemcache.client.rendezvous import RendezvousHash
from pymemcache.exceptions import MemcacheError

logger = logging.getLogger(__name__)

# Return value of ``_begin_server_call`` when the node is temporarily
# unavailable (its retry window has not elapsed yet, or another caller is
# already probing it).
_SKIP = object()

# Return value of ``_begin_server_call`` when the node has exhausted its retry
# attempts and has just been moved to the dead pool. The historical behavior is
# to still attempt the call once against the existing client.
_DEAD = object()


class HashClient:
    """
    A client for communicating with a cluster of memcached servers
    """

    #: :class:`Client` class used to create new clients
    client_class = Client

    def __init__(
        self,
        servers,
        hasher=RendezvousHash,
        serde=None,
        serializer=None,
        deserializer=None,
        connect_timeout=None,
        timeout=None,
        no_delay=False,
        socket_module=socket,
        socket_keepalive=None,
        key_prefix=b"",
        max_pool_size=None,
        pool_idle_timeout=0,
        lock_generator=None,
        retry_attempts=2,
        retry_timeout=1,
        dead_timeout=60,
        use_pooling=False,
        ignore_exc=False,
        allow_unicode_keys=False,
        default_noreply=True,
        encoding="ascii",
        tls_context=None,
    ):
        """
        Constructor.

        Args:
          servers: list() of tuple(hostname, port) or string containing a UNIX
                   socket path.
          hasher: optional class three functions ``get_node``, ``add_node``,
                  and ``remove_node``
                  defaults to Rendezvous (HRW) hash.

          use_pooling: use py:class:`.PooledClient` as the default underlying
                       class. ``max_pool_size`` and ``lock_generator`` can
                       be used with this. default: False

          retry_attempts: Amount of times a client should be tried before it
                          is marked dead and removed from the pool.
          retry_timeout (float): Time in seconds that should pass between retry
                                 attempts.
          dead_timeout (float): Time in seconds before attempting to add a node
                                back in the pool.
          encoding: optional str, controls data encoding (defaults to 'ascii').

        Further arguments are interpreted as for :py:class:`.Client`
        constructor.
        """
        self.clients = {}
        self.retry_attempts = retry_attempts
        self.retry_timeout = retry_timeout
        self.dead_timeout = dead_timeout
        self.use_pooling = use_pooling
        self.key_prefix = key_prefix
        self.ignore_exc = ignore_exc
        self.allow_unicode_keys = allow_unicode_keys
        self._init_failure_state()

        self.hasher = hasher()

        self.default_kwargs = {
            "connect_timeout": connect_timeout,
            "timeout": timeout,
            "no_delay": no_delay,
            "socket_module": socket_module,
            "socket_keepalive": socket_keepalive,
            "key_prefix": key_prefix,
            "serde": serde,
            "serializer": serializer,
            "deserializer": deserializer,
            "allow_unicode_keys": allow_unicode_keys,
            "default_noreply": default_noreply,
            "encoding": encoding,
            "tls_context": tls_context,
        }

        if use_pooling is True:
            self.default_kwargs.update(
                {
                    "max_pool_size": max_pool_size,
                    "pool_idle_timeout": pool_idle_timeout,
                    "lock_generator": lock_generator,
                }
            )

        for server in servers:
            self.add_server(normalize_server_spec(server))
        self.encoding = encoding
        self.tls_context = tls_context

    def _init_failure_state(self):
        """Initialize (or reset) the shared node health state.

        All node health bookkeeping lives behind a single short-lived lock.
        The lock is only held while dictionaries and flags are mutated;
        network requests are always performed without it, so a slow or
        unreachable node cannot block callers talking to healthy nodes.
        """
        self._failed_clients = {}
        self._dead_clients = {}
        self._last_dead_check_time = time.time()
        # Guards ``_failed_clients``, ``_dead_clients`` and the hasher's node
        # membership. It is never held while a network operation is in flight.
        self._state_lock = threading.RLock()

    def _make_client_key(self, server):
        if isinstance(server, (list, tuple)) and len(server) == 2:
            return "%s:%s" % server
        return server

    def add_server(self, server, port=None) -> None:
        # To maintain backward compatibility, if a port is provided, assume
        # that server wasn't provided as a (host, port) tuple.
        if port is not None:
            if not isinstance(server, str):
                raise TypeError("Server must be a string when passing port.")
            server = (server, port)

        _class = PooledClient if self.use_pooling else self.client_class
        client = _class(server, **self.default_kwargs)
        if self.use_pooling:
            client.client_class = self.client_class

        key = self._make_client_key(server)
        with self._state_lock:
            self.clients[key] = client
            self.hasher.add_node(key)

    def _remove_server_locked(self, server, dead_time):
        """Move ``server`` to the dead pool.

        Caller must hold ``_state_lock``.
        """
        key = self._make_client_key(server)
        # Any in-flight recovery attempt for the node is superseded by the
        # dead transition, and the node leaves the failed pool entirely.
        self._failed_clients.pop(server, None)
        self._dead_clients[server] = dead_time
        try:
            self.hasher.remove_node(key)
        except ValueError:
            # The node may already have been removed by a concurrent caller.
            logger.debug("server %s was no longer in rotation", server)

    def remove_server(self, server, port=None) -> None:
        # To maintain backward compatibility, if a port is provided, assume
        # that server wasn't provided as a (host, port) tuple.
        if port is not None:
            if not isinstance(server, str):
                raise TypeError("Server must be a string when passing port.")
            server = (server, port)

        dead_time = time.time()
        with self._state_lock:
            self._remove_server_locked(server, dead_time)

    def _retry_dead(self) -> None:
        current_time = time.time()
        with self._state_lock:
            ldc = self._last_dead_check_time
            # We have reached the retry timeout
            if current_time - ldc <= self.dead_timeout:
                return

            candidates = []
            for server, dead_time in self._dead_clients.items():
                if current_time - dead_time > self.dead_timeout:
                    candidates.append(server)
            for server in candidates:
                logger.debug("bringing server back into rotation %s", server)
                # The node re-enters rotation as a healthy node with a clean
                # slate: any failure metadata left behind by the final
                # pre-dead request is dropped so its retry budget is restored
                # instead of being carried over from the previous incarnation.
                self._failed_clients.pop(server, None)
                # ``add_server`` re-acquires the reentrant lock and puts the
                # node back in the hasher atomically, so concurrent callers
                # can never observe half-finished re-additions.
                self.add_server(server)
                del self._dead_clients[server]
            self._last_dead_check_time = current_time

    def _get_client(self, key):
        # If key is tuple use first item as server key
        if isinstance(key, tuple) and len(key) == 2:
            server_key, key = key
        else:
            server_key = key

        check_key_helper(server_key, self.allow_unicode_keys, self.key_prefix)
        if self._dead_clients:
            self._retry_dead()

        with self._state_lock:
            server = self.hasher.get_node(server_key)
            # We've ran out of servers to try
            if server is None:
                if self.ignore_exc is True:
                    return None, key
                raise MemcacheError("All servers seem to be down right now")

            return self.clients[server], key

    def _begin_server_call(self, server):
        """Decide whether a call to ``server`` may access the network.

        This is the single shared gate for every command, including batch and
        multi-node operations. It returns one of:

        * ``(True, generation)``: the caller owns the node's recovery probe
          and must run the request now, without holding ``_state_lock``.
        * ``(False, _SKIP)``: the node's retry window is still open or another
          caller is already probing it. The node must not be contacted.
        * ``(False, _DEAD)``: the node just exhausted its retry attempts and
          was moved to the dead pool. Historical behavior is preserved by
          still allowing the caller to attempt the request once.
        """
        with self._state_lock:
            failed_metadata = self._failed_clients.get(server)
            if failed_metadata is None:
                # Healthy node: nothing to coordinate, no extra locking is
                # imposed on the actual network request.
                return True, None

            if failed_metadata["attempts"] >= self.retry_attempts:
                # We've reached our max retry attempts, mark the server dead.
                logger.debug("marking server as dead: %s", server)
                self._remove_server_locked(server, time.time())
                return False, _DEAD

            if failed_metadata["probing"]:
                # Another caller is already running the recovery probe; this
                # caller must not touch the node so only one probe happens.
                logger.debug("recovery probe already in progress: %s", server)
                return False, _SKIP

            if time.time() - failed_metadata["failed_time"] <= self.retry_timeout:
                # The retry window has not elapsed yet.
                return False, _SKIP

            # The retry window has elapsed. This caller wins the probe and is
            # solely responsible for re-validating the node.
            logger.debug("retrying failed server: %s", server)
            failed_metadata["probing"] = True
            return True, failed_metadata["generation"]

    def _finish_probe_success(self, server, generation):
        """Clear the failed state after a successful recovery probe.

        The failure is only cleared when no newer failure superseded the
        probe while it was in flight. An earlier request succeeding after a
        newer failure must never resurrect the node.
        """
        with self._state_lock:
            failed_metadata = self._failed_clients.get(server)
            if failed_metadata is None or failed_metadata["generation"] != generation:
                logger.debug(
                    "ignoring stale probe success for server %s", server
                )
                return
            self._failed_clients.pop(server)

    def _release_server_probe(self, server):
        """Release the in-progress probe flag without changing the state.

        Used when a probe fails for a reason that does not mark the node as
        failed (a non-socket exception), so a later caller can probe again.
        """
        with self._state_lock:
            failed_metadata = self._failed_clients.get(server)
            if failed_metadata is not None:
                failed_metadata["probing"] = False

    def _safely_run_func(self, client, func, default_val, *args, **kwargs):
        is_probe = False
        try:
            should_run, probe_generation = self._begin_server_call(client.server)
            if not should_run:
                if probe_generation is _DEAD:
                    # Preserve the historical behavior of attempting the call
                    # once against the just-dead node.
                    return func(*args, **kwargs)
                return default_val

            is_probe = probe_generation is not None
            result = func(*args, **kwargs)

            if is_probe:
                self._finish_probe_success(client.server, probe_generation)
            return result

        # Connecting to the server fail, we should enter
        # retry mode
        except OSError:
            self._mark_failed_server(client.server)

            # if we haven't enabled ignore_exc, don't move on gracefully, just
            # raise the exception
            if not self.ignore_exc:
                raise

            return default_val
        except Exception:
            # any exceptions that aren't socket.error we need to handle
            # gracefully as well
            if is_probe:
                # A non-socket failure does not change the node's state, but
                # the probe slot must be freed so recovery can be retried.
                self._release_server_probe(client.server)
            if not self.ignore_exc:
                raise

            return default_val

    def _safely_run_set_many(self, client, values, *args, **kwargs):
        failed = []
        succeeded = []
        is_probe = False
        try:
            should_run, probe_generation = self._begin_server_call(client.server)
            if not should_run:
                if probe_generation is _DEAD:
                    # Preserve the historical behavior of attempting the call
                    # once against the just-dead node.
                    succeeded, failed, err = self._set_many(
                        client, values, *args, **kwargs
                    )
                    if err is not None:
                        raise err
                    return failed
                return list(values.keys())

            is_probe = probe_generation is not None
            succeeded, failed, err = self._set_many(client, values, *args, **kwargs)
            if err is not None:
                raise err

            if is_probe:
                self._finish_probe_success(client.server, probe_generation)
            return failed

        # Connecting to the server fail, we should enter
        # retry mode
        except OSError:
            self._mark_failed_server(client.server)

            # if we haven't enabled ignore_exc, don't move on gracefully, just
            # raise the exception
            if not self.ignore_exc:
                raise

            return list(set(values.keys()) - set(succeeded))
        except Exception:
            # any exceptions that aren't socket.error we need to handle
            # gracefully as well
            if is_probe:
                self._release_server_probe(client.server)
            if not self.ignore_exc:
                raise

            return list(set(values.keys()) - set(succeeded))

    def _mark_failed_server(self, server):
        with self._state_lock:
            # A node that has already been moved to the dead pool is managed
            # exclusively by the dead reaper; a failure of the legacy
            # one-shot call against it must not resurrect failure metadata
            # (otherwise its retry budget would be polluted on revival).
            if server in self._dead_clients:
                return

            failed_metadata = self._failed_clients.get(server)
            if failed_metadata is None:
                # This client has never failed, lets mark it for failure.
                # ``generation`` identifies this failure epoch: requests that
                # started before it may only clear this exact state.
                self._failed_clients[server] = {
                    "failed_time": time.time(),
                    "attempts": 0,
                    "generation": 1,
                    "probing": False,
                }
                # We aren't allowing any retries, mark the server dead
                # immediately.
                if self.retry_attempts <= 0:
                    logger.debug("marking server as dead %s", server)
                    self._remove_server_locked(server, time.time())
                return

            # This client has failed previously: record another attempt,
            # start a fresh failure epoch, and free any probe slot so the
            # next retry window can re-probe the node.
            failed_metadata["attempts"] += 1
            failed_metadata["failed_time"] = time.time()
            failed_metadata["generation"] += 1
            failed_metadata["probing"] = False

    def _run_cmd(self, cmd, key, default_val, *args, **kwargs):
        client, key = self._get_client(key)

        if client is None:
            return default_val

        func = getattr(client, cmd)
        args = list(args)
        args.insert(0, key)
        return self._safely_run_func(client, func, default_val, *args, **kwargs)

    def _set_many(self, client, values, *args, **kwargs):
        failed = []
        succeeded = []

        try:
            failed = client.set_many(values, *args, **kwargs)
        except Exception as e:
            if not self.ignore_exc:
                return succeeded, failed, e

        succeeded = [key for key in values if key not in failed]
        return succeeded, failed, None

    def close(self):
        # Snapshot the clients: closing a client must not mutate the dict we
        # iterate over, and no state lock is held while network I/O happens.
        for client in list(self.clients.values()):
            self._safely_run_func(client, client.close, False)

    disconnect_all = close

    def set(self, key, *args, **kwargs):
        return self._run_cmd("set", key, False, *args, **kwargs)

    def get(self, key, default=None, **kwargs):
        return self._run_cmd("get", key, default, default=default, **kwargs)

    def gat(self, key, default=None, **kwargs):
        return self._run_cmd("gat", key, default, default=default, **kwargs)

    def gats(self, key, default=None, **kwargs):
        return self._run_cmd("gats", key, default, default=default, **kwargs)

    def incr(self, key, *args, **kwargs):
        return self._run_cmd("incr", key, None, *args, **kwargs)

    def decr(self, key, *args, **kwargs):
        return self._run_cmd("decr", key, None, *args, **kwargs)

    def set_many(self, values, *args, **kwargs):
        client_batches = collections.defaultdict(dict)
        failed = []

        for key, value in values.items():
            client, key = self._get_client(key)

            if client is None:
                failed.append(key)
                continue

            client_batches[client.server][key] = value

        for server, values in client_batches.items():
            client = self.clients[self._make_client_key(server)]
            failed += self._safely_run_set_many(client, values, *args, **kwargs)

        return failed

    set_multi = set_many

    def get_many(self, keys, gets=False, *args, **kwargs):
        client_batches = collections.defaultdict(list)
        end = {}

        for key in keys:
            client, key = self._get_client(key)

            if client is None:
                continue

            client_batches[client.server].append(key)

        for server, keys in client_batches.items():
            client = self.clients[self._make_client_key(server)]
            new_args = list(args)
            new_args.insert(0, keys)

            if gets:
                get_func = client.gets_many
            else:
                get_func = client.get_many

            result = self._safely_run_func(client, get_func, {}, *new_args, **kwargs)
            end.update(result)

        return end

    get_multi = get_many

    def gets(self, key, *args, **kwargs):
        return self._run_cmd("gets", key, None, *args, **kwargs)

    def gets_many(self, keys, *args, **kwargs):
        return self.get_many(keys, gets=True, *args, **kwargs)

    gets_multi = gets_many

    def add(self, key, *args, **kwargs):
        return self._run_cmd("add", key, False, *args, **kwargs)

    def prepend(self, key, *args, **kwargs):
        return self._run_cmd("prepend", key, False, *args, **kwargs)

    def append(self, key, *args, **kwargs):
        return self._run_cmd("append", key, False, *args, **kwargs)

    def delete(self, key, *args, **kwargs):
        return self._run_cmd("delete", key, False, *args, **kwargs)

    def delete_many(self, keys, *args, **kwargs) -> bool:
        for key in keys:
            self._run_cmd("delete", key, False, *args, **kwargs)
        return True

    delete_multi = delete_many

    def cas(self, key, *args, **kwargs):
        return self._run_cmd("cas", key, False, *args, **kwargs)

    def replace(self, key, *args, **kwargs):
        return self._run_cmd("replace", key, False, *args, **kwargs)

    def touch(self, key, *args, **kwargs):
        return self._run_cmd("touch", key, False, *args, **kwargs)

    def stats(self, *args, **kwargs):
        result = list()
        for key, client in list(self.clients.items()):
            result.append(
                (
                    key,
                    self._safely_run_func(client, client.stats, False, *args, **kwargs),
                )
            )
        return result

    def flush_all(self, *args, **kwargs) -> None:
        for client in list(self.clients.values()):
            self._safely_run_func(client, client.flush_all, False, *args, **kwargs)

    def quit(self) -> None:
        for client in list(self.clients.values()):
            self._safely_run_func(client, client.quit, False)
