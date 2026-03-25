import time
import typing
import threading
import copy
import asyncio
import concurrent.futures
import hashlib
import json
import uuid
import bittensor as bt

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

# Bittensor Miner Template:
import talisman_ai

# import base miner class which takes care of most of the boilerplate
from talisman_ai.base.miner import BaseMinerNeuron
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer import setup_telegram_analyzer
from talisman_ai.utils.api_models import TweetAnalysisBase, TelegramMessageAnalysis


class Miner(BaseMinerNeuron):
    """
    V3 Miner: Processes TweetBatch requests from validators.
    
    The miner receives batches of tweets from validators, analyzes each tweet
    for subnet relevance and sentiment, and returns the enriched batch.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Initialize analyzer for tweet classification
        bt.logging.info("[Miner] Initializing analyzer...")
        self.analyzer = setup_analyzer()
        self.telegram_analyzer = setup_telegram_analyzer()
        bt.logging.info("[Miner] Analyzer initialized")
        # NOTE: we intentionally do NOT reuse a single bt.Dendrite across threads/event-loops.
        # Miner responses are sent back to validators from a background thread with its own event loop.

        # Use a bounded executor instead of spawning an unbounded thread per request.
        # This protects the miner from overload (thread explosion) which otherwise leads to timeouts and penalties.
        max_workers = int(getattr(talisman_ai.config, "MINER_WORKERS", 8))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="miner_worker_",
        )
        self._max_pending_tasks = max(
            max(1, max_workers),
            int(getattr(talisman_ai.config, "MINER_MAX_PENDING_TASKS", max_workers * 16)),
        )
        self._cache_enabled = str(getattr(talisman_ai.config, "MINER_CACHE_ENABLED", "true")).lower() in ("1", "true", "yes", "on")
        self._cache_ttl_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_TTL_SECONDS", 1800.0))
        self._cache_max_items = max(1, int(getattr(talisman_ai.config, "MINER_CACHE_MAX_ITEMS", 10000)))
        self._cache_log_interval_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_LOG_INTERVAL_SECONDS", 60.0))
        self._cache_backend = str(getattr(talisman_ai.config, "MINER_CACHE_BACKEND", "local")).lower()
        self._tweet_analysis_cache: dict = {}
        self._telegram_analysis_cache: dict = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_last_log_ts = time.time()
        self._inflight_lock = threading.Lock()
        self._tweet_inflight: dict = {}
        self._telegram_inflight: dict = {}
        self._cache_inflight_joins = 0
        self._pending_lock = threading.Lock()
        self._pending_tasks = 0
        self._pending_rejections = 0

        # Redis distributed cache backend (shared across multiple miner processes on same server)
        self._redis_enabled = self._cache_backend == "redis" and redis_lib is not None
        self._redis = None
        base_namespace = str(getattr(talisman_ai.config, "MINER_CACHE_REDIS_NAMESPACE", "talisman:sn45"))
        analysis_sig_src = f"{getattr(talisman_ai.config, 'MODEL', '')}|{getattr(talisman_ai.config, 'LLM_BASE', '')}"
        analysis_sig = hashlib.sha256(analysis_sig_src.encode("utf-8")).hexdigest()[:12]
        self._redis_namespace = f"{base_namespace}:{analysis_sig}"
        self._redis_lock_ttl_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_LOCK_TTL_SECONDS", 60.0))
        self._redis_wait_timeout_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_WAIT_TIMEOUT_SECONDS", 10.0))
        self._redis_wait_poll_interval_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_WAIT_POLL_INTERVAL_SECONDS", 0.2))
        self._redis_lock_heartbeat_seconds = float(
            getattr(
                talisman_ai.config,
                "MINER_CACHE_LOCK_HEARTBEAT_SECONDS",
                max(1.0, self._redis_lock_ttl_seconds / 3.0),
            )
        )
        self._cache_hits_redis = 0
        self._cache_misses_redis = 0
        self._cache_wait_hits_redis = 0
        self._cache_computed_redis = 0

        if self._redis_enabled:
            try:
                redis_url = str(getattr(talisman_ai.config, "REDIS_URL", "redis://127.0.0.1:6379/0"))
                redis_password = getattr(talisman_ai.config, "REDIS_PASSWORD", None)
                if redis_password:
                    self._redis = redis_lib.Redis.from_url(redis_url, password=redis_password, decode_responses=False)
                else:
                    self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=False)
                # Basic connectivity check
                self._redis.ping()
                bt.logging.info(f"[Miner][Cache] Redis backend enabled: namespace={self._redis_namespace}")
            except Exception as e:
                bt.logging.warning(f"[Miner][Cache] Redis backend disabled (init failed): {e}")
                self._redis_enabled = False
                self._redis = None

        # IMPORTANT: Register a concrete TweetBatch handler on the axon.
        # Bittensor routes requests by synapse class name; attaching only `forward(self, bt.Synapse)`
        # registers the generic `Synapse` endpoint and does *not* register `TweetBatch`.
        self.axon.attach(
            forward_fn=self.forward_tweets,
            blacklist_fn=self.blacklist_tweet_batch,
            priority_fn=self.priority_tweet_batch,
        )
        
        # Register TelegramBatch handler
        self.axon.attach(
            forward_fn=self.forward_telegram_messages,
            blacklist_fn=self.blacklist_telegram_batch,
            priority_fn=self.priority_telegram_batch,
        )
        
        hotkey = self.wallet.hotkey.ss58_address
        bt.logging.info(f"[Miner] V3 miner started with hotkey: {hotkey}")

    def _make_tweet_cache_key(self, tweet) -> str:
        tweet_id = str(getattr(tweet, "id", "") or "")
        text = str(getattr(tweet, "text", "") or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"tweet:{tweet_id}:{digest}"

    def _make_telegram_cache_key(self, msg) -> str:
        message_id = str(getattr(msg, "id", "") or "")
        content = str(getattr(msg, "content", "") or "")
        inherited_subnet_id = str(getattr(msg, "inherited_subnet_id", "") or "")
        context_messages = getattr(msg, "context_messages", None) or []
        ctx_parts = []
        for ctx in context_messages:
            ctx_id = str(getattr(ctx, "id", "") or "")
            ctx_content = str(getattr(ctx, "content", "") or "")
            ctx_parts.append(f"{ctx_id}:{ctx_content}")
        ctx_blob = "|".join(ctx_parts)
        digest = hashlib.sha256(f"{content}|{ctx_blob}|{inherited_subnet_id}".encode("utf-8")).hexdigest()
        return f"telegram:{message_id}:{digest}"

    def _cache_get(self, cache: dict, key: str):
        if not self._cache_enabled:
            return None
        now = time.time()
        with self._cache_lock:
            entry = cache.get(key)
            if not entry:
                self._cache_misses += 1
                return None
            expires_at, payload = entry
            if expires_at < now:
                try:
                    del cache[key]
                except Exception:
                    pass
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            return payload

    def _cache_set(self, cache: dict, key: str, payload: dict) -> None:
        if not self._cache_enabled:
            return
        now = time.time()
        expires_at = now + self._cache_ttl_seconds
        with self._cache_lock:
            cache[key] = (expires_at, payload)
            if len(cache) > self._cache_max_items:
                # Remove oldest entries first (insertion-ordered dict on modern Python).
                overflow = len(cache) - self._cache_max_items
                for _ in range(max(0, overflow)):
                    try:
                        cache.pop(next(iter(cache)))
                    except Exception:
                        break

    def _redis_data_key(self, local_cache_key: str) -> str:
        # local_cache_key already encodes tweet/message identity + content hash.
        return f"{self._redis_namespace}:{local_cache_key}"

    def _redis_lock_key(self, data_key: str) -> str:
        digest = hashlib.sha256(data_key.encode("utf-8")).hexdigest()
        return f"{self._redis_namespace}:lock:{digest}"

    def _redis_get_payload(self, data_key: str):
        if not self._redis_enabled or self._redis is None:
            return None
        try:
            raw = self._redis.get(data_key)
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except Exception:
            return None

    def _redis_set_payload(self, data_key: str, payload: dict) -> None:
        if not self._redis_enabled or self._redis is None:
            return
        try:
            dumped = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            # Use at least 1 second for EX.
            ex = max(1, int(self._cache_ttl_seconds))
            self._redis.set(data_key, dumped, ex=ex)
        except Exception:
            pass

    def _redis_acquire_lock(self, lock_key: str, token: str) -> bool:
        if not self._redis_enabled or self._redis is None:
            return False
        try:
            # SET key value NX EX ttl
            ex = max(1, int(self._redis_lock_ttl_seconds))
            ok = self._redis.set(lock_key, token, nx=True, ex=ex)
            return bool(ok)
        except Exception:
            return False

    def _redis_release_lock(self, lock_key: str, token: str) -> None:
        if not self._redis_enabled or self._redis is None:
            return
        try:
            # Release lock only if still owned (token matches)
            lua = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """
            self._redis.eval(lua, 1, lock_key, token)
        except Exception:
            pass

    def _redis_refresh_lock(self, lock_key: str, token: str) -> bool:
        if not self._redis_enabled or self._redis is None:
            return False
        try:
            ttl_ms = max(1000, int(self._redis_lock_ttl_seconds * 1000))
            lua = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('pexpire', KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            result = self._redis.eval(lua, 1, lock_key, token, ttl_ms)
            return bool(result)
        except Exception:
            return False

    def _start_lock_heartbeat(self, lock_key: str, token: str):
        stop_event = threading.Event()

        def _heartbeat():
            interval = max(0.25, self._redis_lock_heartbeat_seconds)
            while not stop_event.wait(interval):
                if not self._redis_refresh_lock(lock_key, token):
                    break

        thread = threading.Thread(
            target=_heartbeat,
            daemon=True,
            name="miner_redis_lock_heartbeat",
        )
        thread.start()
        return stop_event, thread

    def _distributed_get_or_compute(
        self,
        local_cache_key: str,
        compute_fn: typing.Callable[[], dict],
        kind_label: str,
        validator_hotkey: str,
    ) -> dict:
        """
        Distributed (Redis) singleflight-style cache:
        L1: local dict (already checked outside)
        L2: Redis GET
        L3: Redis lock + wait/poll to prevent stampede
        """
        data_key = self._redis_data_key(local_cache_key)
        payload = self._redis_get_payload(data_key)
        if payload is not None:
            with self._cache_lock:
                self._cache_hits_redis += 1
            bt.logging.info(
                f"[Miner][Cache][Redis] {kind_label} redis hit local_key_prefix={local_cache_key[:28]}.. validator={validator_hotkey[:12]}.."
            )
            return payload

        with self._cache_lock:
            self._cache_misses_redis += 1

        lock_key = self._redis_lock_key(data_key)
        token = uuid.uuid4().hex
        lock_acquired = self._redis_acquire_lock(lock_key, token)
        if lock_acquired:
            heartbeat_stop = None
            heartbeat_thread = None
            try:
                heartbeat_stop, heartbeat_thread = self._start_lock_heartbeat(lock_key, token)
                # Re-check after acquiring lock (another process may have filled it).
                payload = self._redis_get_payload(data_key)
                if payload is not None:
                    with self._cache_lock:
                        self._cache_hits_redis += 1
                    bt.logging.info(
                        f"[Miner][Cache][Redis] {kind_label} redis hit-after-lock validator={validator_hotkey[:12]}.."
                    )
                    return payload

                payload = compute_fn()
                self._redis_set_payload(data_key, payload)
                with self._cache_lock:
                    self._cache_computed_redis += 1
                bt.logging.info(
                    f"[Miner][Cache][Redis] {kind_label} computed by lock-holder validator={validator_hotkey[:12]}.."
                )
                return payload
            finally:
                if heartbeat_stop is not None:
                    heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1.0)
                self._redis_release_lock(lock_key, token)
        else:
            # Wait briefly for the lock holder to populate Redis.
            deadline = time.time() + self._redis_wait_timeout_seconds
            while time.time() < deadline:
                payload = self._redis_get_payload(data_key)
                if payload is not None:
                    with self._cache_lock:
                        self._cache_wait_hits_redis += 1
                    bt.logging.info(
                        f"[Miner][Cache][Redis] {kind_label} wait-hit validator={validator_hotkey[:12]}.."
                    )
                    return payload
                time.sleep(self._redis_wait_poll_interval_seconds)

            # Last resort: compute anyway to avoid long stalls.
            payload = compute_fn()
            self._redis_set_payload(data_key, payload)
            with self._cache_lock:
                self._cache_computed_redis += 1
            bt.logging.info(
                f"[Miner][Cache][Redis] {kind_label} computed after-wait-timeout validator={validator_hotkey[:12]}.."
            )
            return payload

    def _get_or_compute_payload(
        self,
        *,
        local_cache: dict,
        inflight_map: dict,
        cache_key: str,
        compute_fn: typing.Callable[[], dict],
        kind_label: str,
        validator_hotkey: str,
    ) -> dict:
        cached = self._cache_get(local_cache, cache_key)
        if cached is not None:
            bt.logging.info(
                f"[Miner][Cache] {kind_label} local cache hit key_prefix={cache_key[:28]}.. "
                f"validator={validator_hotkey[:12]}.."
            )
            return cached

        owner_future = None
        is_owner = False
        with self._inflight_lock:
            owner_future = inflight_map.get(cache_key)
            if owner_future is None:
                owner_future = concurrent.futures.Future()
                inflight_map[cache_key] = owner_future
                is_owner = True
            else:
                with self._cache_lock:
                    self._cache_inflight_joins += 1

        if not is_owner:
            bt.logging.info(
                f"[Miner][Cache] {kind_label} inflight-join key_prefix={cache_key[:28]}.. "
                f"validator={validator_hotkey[:12]}.."
            )
            payload = owner_future.result()
            self._cache_set(local_cache, cache_key, payload)
            return payload

        try:
            cached = self._cache_get(local_cache, cache_key)
            if cached is not None:
                owner_future.set_result(cached)
                return cached

            if self._redis_enabled:
                payload = self._distributed_get_or_compute(
                    local_cache_key=cache_key,
                    compute_fn=compute_fn,
                    kind_label=kind_label,
                    validator_hotkey=validator_hotkey,
                )
            else:
                payload = compute_fn()
            self._cache_set(local_cache, cache_key, payload)
            owner_future.set_result(payload)
            return payload
        except Exception as e:
            owner_future.set_exception(e)
            raise
        finally:
            with self._inflight_lock:
                inflight_map.pop(cache_key, None)

    def _submit_background_task(self, fn: typing.Callable, *args) -> bool:
        with self._pending_lock:
            if self._pending_tasks >= self._max_pending_tasks:
                self._pending_rejections += 1
                return False
            self._pending_tasks += 1

        try:
            future = self._executor.submit(fn, *args)
        except Exception:
            with self._pending_lock:
                self._pending_tasks = max(0, self._pending_tasks - 1)
            raise

        future.add_done_callback(self._on_background_task_done)
        return True

    def _on_background_task_done(self, future: concurrent.futures.Future) -> None:
        with self._pending_lock:
            self._pending_tasks = max(0, self._pending_tasks - 1)
        try:
            future.result()
        except Exception as e:
            bt.logging.error(f"[Miner] Background task failed: {e}")

    def _maybe_log_cache_stats(self) -> None:
        if not self._cache_enabled:
            return
        now = time.time()
        with self._cache_lock:
            if (now - self._cache_last_log_ts) < self._cache_log_interval_seconds:
                return
            hits = int(self._cache_hits)
            misses = int(self._cache_misses)
            total = hits + misses
            hit_rate = (hits / total * 100.0) if total > 0 else 0.0
            tweet_size = len(self._tweet_analysis_cache)
            telegram_size = len(self._telegram_analysis_cache)
            redis_hits = int(self._cache_hits_redis)
            redis_misses = int(self._cache_misses_redis)
            redis_wait_hits = int(self._cache_wait_hits_redis)
            redis_computed = int(self._cache_computed_redis)
            inflight_joins = int(self._cache_inflight_joins)
            with self._pending_lock:
                pending_tasks = int(self._pending_tasks)
                pending_rejections = int(self._pending_rejections)
            self._cache_last_log_ts = now

        bt.logging.info(
            f"[Miner][CacheStats] hits={hits} misses={misses} "
            f"hit_rate={hit_rate:.2f}% tweet_cache={tweet_size} "
            f"telegram_cache={telegram_size} ttl_s={self._cache_ttl_seconds:.0f} "
            f"redis_hits={redis_hits} redis_misses={redis_misses} redis_wait_hits={redis_wait_hits} "
            f"redis_computed={redis_computed} inflight_joins={inflight_joins} "
            f"pending_tasks={pending_tasks}/{self._max_pending_tasks} pending_rejections={pending_rejections}"
        )

    async def blacklist_tweet_batch(
        self, synapse: talisman_ai.protocol.TweetBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.blacklist(synapse)

    async def priority_tweet_batch(self, synapse: talisman_ai.protocol.TweetBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.priority(synapse)

    async def blacklist_telegram_batch(
        self, synapse: talisman_ai.protocol.TelegramBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TelegramBatch."""
        return await self.blacklist(synapse)

    async def priority_telegram_batch(self, synapse: talisman_ai.protocol.TelegramBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TelegramBatch."""
        return await self.priority(synapse)
    
    async def forward_is_alive(self, synapse: talisman_ai.protocol.IsAlive) -> talisman_ai.protocol.IsAlive:
        """
        Processes incoming IsAlive synapses from validators.
        """
        synapse.is_alive = True
        return synapse
    
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Processes incoming synapses. Routes TweetBatch requests to forward_tweets.
        
        Args:
            synapse (bt.Synapse): The incoming synapse request.
            
        Returns:
            bt.Synapse: The processed synapse response.
        """
        if isinstance(synapse, talisman_ai.protocol.TweetBatch):
            return await self.forward_tweets(synapse)
        
        bt.logging.warning(f"Received synapse type: {type(synapse).__name__}, but no handler implemented")
        return synapse

    async def forward_tweets(self, synapse: talisman_ai.protocol.TweetBatch) -> talisman_ai.protocol.TweetBatch:
        """
        Processes TweetBatch requests from validators.
        
        Spawns a background thread to analyze tweets and send results back to the validator.
        Returns immediately to avoid blocking the axon.
        
        Args:
            synapse: TweetBatch containing list of tweets to analyze
            
        Returns:
            TweetBatch (returns immediately, processing happens in background)
        """
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from validator {validator_hotkey}")
        
        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse
        
        # Make a deep copy of the synapse for background processing
        synapse_copy = copy.deepcopy(synapse)
        
        # Process asynchronously in a bounded worker pool.
        submitted = self._submit_background_task(self._process_and_send_tweets, synapse_copy, validator_hotkey)
        if not submitted:
            bt.logging.warning(
                f"[Miner] Background queue full ({self._max_pending_tasks}); "
                f"dropping TweetBatch from validator {validator_hotkey[:12]}.."
            )
            self._maybe_log_cache_stats()
            return synapse
        
        bt.logging.info(f"[Miner] Started background processing for TweetBatch, returning immediately")
        return synapse

    async def forward_telegram_messages(self, synapse: talisman_ai.protocol.TelegramBatch) -> talisman_ai.protocol.TelegramBatch:
        """
        Processes TelegramBatch requests from validators.
        
        Spawns a background thread to analyze telegram messages and send results back to the validator.
        Returns immediately to avoid blocking the axon.
        
        Args:
            synapse: TelegramBatch containing list of messages to analyze
            
        Returns:
            TelegramBatch (returns immediately, processing happens in background)
        """
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received TelegramBatch with {len(synapse.message_batch)} message(s) from validator {validator_hotkey}")
        
        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse
        
        # Make a deep copy of the synapse for background processing
        synapse_copy = copy.deepcopy(synapse)
        
        # Process asynchronously in a bounded worker pool.
        submitted = self._submit_background_task(self._process_and_send_telegram_messages, synapse_copy, validator_hotkey)
        if not submitted:
            bt.logging.warning(
                f"[Miner] Background queue full ({self._max_pending_tasks}); "
                f"dropping TelegramBatch from validator {validator_hotkey[:12]}.."
            )
            self._maybe_log_cache_stats()
            return synapse
        
        bt.logging.info(f"[Miner] Started background processing for TelegramBatch, returning immediately")
        return synapse

    def _process_and_send_tweets(self, synapse: talisman_ai.protocol.TweetBatch, validator_hotkey: str):
        """
        Background thread function to process tweets and send results back to validator.
        
        Args:
            synapse: TweetBatch to process
            validator_hotkey: Hotkey of the validator to send results back to
        """
        try:
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.tweet_batch)} tweets")
            
            # Process each tweet (always attach analysis; missing analysis is frequently sampled and rejected)
            for tweet in synapse.tweet_batch:
                text = tweet.text or ""
                cache_key = self._make_tweet_cache_key(tweet)

                def compute_tweet_payload() -> dict:
                    # Classify the tweet (validator will re-run the same analyzer on the same text)
                    classification = None
                    try:
                        classification = self.analyzer.classify_post(text)
                    except Exception:
                        classification = None

                    # If classification fails, fall back to defaults to avoid returning missing analysis.
                    if classification is None:
                        bt.logging.warning(
                            f"[Miner] Failed to classify tweet {tweet.id}; returning fallback analysis"
                        )
                        return {
                            "sentiment": "neutral",
                            "subnet_id": 0,
                            "subnet_name": "NONE_OF_THE_ABOVE",
                            "content_type": "other",
                            "technical_quality": "none",
                            "market_analysis": "other",
                            "impact_potential": "NONE",
                        }

                    # Create payload with required fields for validator.
                    return {
                        "sentiment": classification.sentiment.value,
                        "subnet_id": classification.subnet_id,
                        "subnet_name": classification.subnet_name,
                        "content_type": classification.content_type.value,
                        "technical_quality": classification.technical_quality.value,
                        "market_analysis": classification.market_analysis.value,
                        "impact_potential": classification.impact_potential.value,
                    }

                payload = self._get_or_compute_payload(
                    local_cache=self._tweet_analysis_cache,
                    inflight_map=self._tweet_inflight,
                    cache_key=cache_key,
                    compute_fn=compute_tweet_payload,
                    kind_label="Tweet",
                    validator_hotkey=validator_hotkey,
                )
                tweet.analysis = TweetAnalysisBase(**payload)
                self._maybe_log_cache_stats()
            
            bt.logging.info(f"[Miner] Background: Finished processing, sending back to validator {validator_hotkey}")
            
            # Find validator UID and axon info from metagraph
            try:
                validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
            except ValueError:
                bt.logging.error(f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph")
                return
            
            validator_axon = self.metagraph.axons[validator_uid]
            bt.logging.info(f"[Miner] Background: Found validator UID {validator_uid}, sending response via dendrite")
            
            # Send the processed batch back to the validator.
            push_timeout = float(getattr(talisman_ai.config, "MINER_PUSH_TIMEOUT", 90.0))
            max_retries = max(0, int(getattr(talisman_ai.config, "MINER_PUSH_RETRIES", 1)))
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            dendrite = None
            try:
                for attempt in range(max_retries + 1):
                    if dendrite is not None:
                        try:
                            if hasattr(dendrite, "aclose_session"):
                                loop.run_until_complete(dendrite.aclose_session())
                            elif hasattr(dendrite, "close_session"):
                                dendrite.close_session()
                        except Exception:
                            pass
                        dendrite = None
                    dendrite = bt.Dendrite(wallet=self.wallet)
                    responses = loop.run_until_complete(
                        dendrite.forward(
                            axons=[validator_axon],
                            synapse=synapse,
                            timeout=push_timeout,
                            deserialize=True,
                        )
                    )
                    try:
                        status_code = responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                        status_msg = responses[0].dendrite.status_message if responses and responses[0].dendrite else None
                    except Exception:
                        status_code, status_msg = None, None

                    if status_code == 200:
                        bt.logging.info(
                            f"[Miner] Background: Successfully sent processed TweetBatch back to validator {validator_hotkey}"
                        )
                        break

                    retryable = status_code in (408, 429, 500, 502, 503, 504) or status_code is None
                    if retryable and attempt < max_retries:
                        bt.logging.warning(
                            f"[Miner] Background: TweetBatch push failed (status={status_code}): {status_msg}; "
                            f"retrying {attempt + 1}/{max_retries} after backoff"
                        )
                        time.sleep(2.0 * (attempt + 1))
                        continue

                    bt.logging.error(
                        f"[Miner] Background: Validator response failed (status={status_code}): {status_msg}"
                    )
                    break
            except Exception as e:
                bt.logging.error(f"[Miner] Background: Failed to send response to validator: {e}")
            finally:
                try:
                    if dendrite is not None:
                        if hasattr(dendrite, "aclose_session"):
                            loop.run_until_complete(dendrite.aclose_session())
                        elif hasattr(dendrite, "close_session"):
                            dendrite.close_session()
                except Exception:
                    pass
                loop.close()
                
        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing tweets: {e}")

    def _process_and_send_telegram_messages(self, synapse: talisman_ai.protocol.TelegramBatch, validator_hotkey: str):
        """
        Background thread function to process telegram messages and send results back to validator.
        
        Args:
            synapse: TelegramBatch to process
            validator_hotkey: Hotkey of the validator to send results back to
        """
        try:
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.message_batch)} telegram messages")
            
            # Process each message (always attach analysis; missing analysis is frequently sampled and rejected)
            for msg in synapse.message_batch:
                content = msg.content or ""
                cache_key = self._make_telegram_cache_key(msg)
                
                # Build message dict for analyzer
                messages_for_analysis = [{
                    'message_id': msg.id,
                    'username': msg.sender_username or msg.sender_name,
                    'content': content,
                }]
                
                # Add context messages if available
                if msg.context_messages:
                    for ctx in msg.context_messages:
                        messages_for_analysis.insert(0, {
                            'message_id': ctx.id,
                            'username': ctx.sender_username or ctx.sender_name,
                            'content': ctx.content,
                        })
                
                # Use inherited subnet_id if provided (don't reclassify)
                inherited_subnet_id = msg.inherited_subnet_id

                from datetime import datetime

                def compute_telegram_payload() -> dict:
                    # Classify the message group
                    classification = None
                    try:
                        classification = self.telegram_analyzer.classify_message_group(
                            messages_for_analysis,
                            subnet_id=inherited_subnet_id,
                        )
                    except Exception:
                        classification = None

                    if classification is None:
                        bt.logging.warning(
                            f"[Miner] Failed to classify telegram message {msg.id}; returning fallback analysis"
                        )
                        return {
                            "id": 0,
                            "message_id": msg.id,
                            "sentiment": "neutral",
                            "subnet_id": int(inherited_subnet_id) if inherited_subnet_id is not None else 0,
                            "subnet_name": "NONE_OF_THE_ABOVE",
                            "content_type": "other",
                            "technical_quality": "none",
                            "market_analysis": "other",
                            "impact_potential": "NONE",
                            "relevance_confidence": None,
                            "analyzed_at": datetime.now().isoformat(),
                        }

                    return {
                        "id": 0,
                        "message_id": msg.id,
                        "sentiment": classification.sentiment.value,
                        "subnet_id": classification.subnet_id,
                        "subnet_name": classification.subnet_name,
                        "content_type": classification.content_type.value,
                        "technical_quality": classification.technical_quality.value,
                        "market_analysis": classification.market_analysis.value,
                        "impact_potential": classification.impact_potential.value,
                        "relevance_confidence": classification.relevance_confidence,
                        "analyzed_at": datetime.now().isoformat(),
                    }

                payload = self._get_or_compute_payload(
                    local_cache=self._telegram_analysis_cache,
                    inflight_map=self._telegram_inflight,
                    cache_key=cache_key,
                    compute_fn=compute_telegram_payload,
                    kind_label="Telegram",
                    validator_hotkey=validator_hotkey,
                )
                msg.analysis = TelegramMessageAnalysis(**payload)
                self._maybe_log_cache_stats()
            
            bt.logging.info(f"[Miner] Background: Finished processing telegram messages, sending back to validator {validator_hotkey}")
            
            # Find validator UID and axon info from metagraph
            try:
                validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
            except ValueError:
                bt.logging.error(f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph")
                return
            
            validator_axon = self.metagraph.axons[validator_uid]
            bt.logging.info(f"[Miner] Background: Found validator UID {validator_uid}, sending telegram response via dendrite")
            
            # Send the processed batch back to the validator.
            push_timeout = float(getattr(talisman_ai.config, "MINER_PUSH_TIMEOUT", 90.0))
            max_retries = max(0, int(getattr(talisman_ai.config, "MINER_PUSH_RETRIES", 1)))
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            dendrite = None
            try:
                for attempt in range(max_retries + 1):
                    if dendrite is not None:
                        try:
                            if hasattr(dendrite, "aclose_session"):
                                loop.run_until_complete(dendrite.aclose_session())
                            elif hasattr(dendrite, "close_session"):
                                dendrite.close_session()
                        except Exception:
                            pass
                        dendrite = None
                    dendrite = bt.Dendrite(wallet=self.wallet)
                    responses = loop.run_until_complete(
                        dendrite.forward(
                            axons=[validator_axon],
                            synapse=synapse,
                            timeout=push_timeout,
                            deserialize=True,
                        )
                    )
                    try:
                        status_code = responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                        status_msg = responses[0].dendrite.status_message if responses and responses[0].dendrite else None
                    except Exception:
                        status_code, status_msg = None, None

                    if status_code == 200:
                        bt.logging.info(
                            f"[Miner] Background: Successfully sent processed TelegramBatch back to validator {validator_hotkey}"
                        )
                        break

                    retryable = status_code in (408, 429, 500, 502, 503, 504) or status_code is None
                    if retryable and attempt < max_retries:
                        bt.logging.warning(
                            f"[Miner] Background: TelegramBatch push failed (status={status_code}): {status_msg}; "
                            f"retrying {attempt + 1}/{max_retries} after backoff"
                        )
                        time.sleep(2.0 * (attempt + 1))
                        continue

                    bt.logging.error(
                        f"[Miner] Background: Validator response failed (status={status_code}): {status_msg}"
                    )
                    break
            except Exception as e:
                bt.logging.error(f"[Miner] Background: Failed to send telegram response to validator: {e}")
            finally:
                try:
                    if dendrite is not None:
                        if hasattr(dendrite, "aclose_session"):
                            loop.run_until_complete(dendrite.aclose_session())
                        elif hasattr(dendrite, "close_session"):
                            dendrite.close_session()
                except Exception:
                    pass
                loop.close()
                
        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing telegram messages: {e}")

    async def forward_score(self, synapse: talisman_ai.protocol.Score) -> talisman_ai.protocol.Score:
        """
        Processes incoming Score synapses from validators.
        
        Receives the score that the validator has given this hotkey for a 100-block interval.
        """
        block_window_start = synapse.block_window_start
        block_window_end = synapse.block_window_end
        score = synapse.score
        validator_hotkey = synapse.validator_hotkey
        bt.logging.info(
            f"[Score] Received score: {score:.6f} from validator {validator_hotkey} for block window {block_window_start}-{block_window_end}"
        )
        return synapse

    async def forward_validation_result(self, synapse: talisman_ai.protocol.ValidationResult) -> talisman_ai.protocol.ValidationResult:
        """
        Processes incoming ValidationResult synapses from validators.
        
        Receives validation results for a specific post, including whether it passed or failed and why.
        """
        validation_id = synapse.validation_id
        post_id = synapse.post_id
        success = synapse.success
        validator_hotkey = synapse.validator_hotkey
        failure_reason = synapse.failure_reason
        
        if success:
            bt.logging.info(
                f"[ValidationResult] ✓ Post {post_id} PASSED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id})"
            )
        else:
            failure_code = failure_reason.get("code", "unknown") if failure_reason else "unknown"
            failure_message = failure_reason.get("message", "Unknown error") if failure_reason else "Unknown error"
            bt.logging.warning(
                f"[ValidationResult] ✗ Post {post_id} FAILED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id}): {failure_code} - {failure_message}"
            )
        
        return synapse

    async def blacklist(
        self, synapse: bt.Synapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bt.Synapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        # Check if hotkey is registered BEFORE trying to get its index
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Only get uid if hotkey is in metagraph (to avoid IndexError)
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Hotkey not found in metagraph (shouldn't happen if check above passed, but be safe)
            bt.logging.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            return True, "Hotkey not in metagraph"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bt.Synapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up when miner exits."""
        try:
            if hasattr(self, "_executor") and self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().__exit__(exc_type, exc_value, traceback)
        return False


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)

