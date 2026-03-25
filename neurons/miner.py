import time
import typing
import threading
import copy
import asyncio
import concurrent.futures
import hashlib
import bittensor as bt

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
        self._cache_enabled = str(getattr(talisman_ai.config, "MINER_CACHE_ENABLED", "true")).lower() in ("1", "true", "yes", "on")
        self._cache_ttl_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_TTL_SECONDS", 1800.0))
        self._cache_max_items = max(1, int(getattr(talisman_ai.config, "MINER_CACHE_MAX_ITEMS", 10000)))
        self._cache_log_interval_seconds = float(getattr(talisman_ai.config, "MINER_CACHE_LOG_INTERVAL_SECONDS", 60.0))
        self._tweet_analysis_cache: dict = {}
        self._telegram_analysis_cache: dict = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_last_log_ts = time.time()

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
            self._cache_last_log_ts = now

        bt.logging.info(
            "[Miner][CacheStats] hits=%d misses=%d hit_rate=%.2f%% tweet_cache=%d telegram_cache=%d ttl_s=%.0f",
            hits, misses, hit_rate, tweet_size, telegram_size, self._cache_ttl_seconds
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
        self._executor.submit(self._process_and_send_tweets, synapse_copy, validator_hotkey)
        
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
        self._executor.submit(self._process_and_send_telegram_messages, synapse_copy, validator_hotkey)
        
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
                cached = self._cache_get(self._tweet_analysis_cache, cache_key)
                if cached is not None:
                    bt.logging.info(
                        f"[Miner][Cache] Tweet cache hit for tweet_id={tweet.id} from validator={validator_hotkey[:12]}.."
                    )
                    tweet.analysis = TweetAnalysisBase(**cached)
                    self._maybe_log_cache_stats()
                    continue

                # Classify the tweet (validator will re-run the same analyzer on the same text)
                classification = None
                try:
                    classification = self.analyzer.classify_post(text)
                except Exception:
                    classification = None
                
                # If classification fails, fall back to analyzer defaults to avoid returning missing analysis.
                if classification is None:
                    bt.logging.warning(f"[Miner] Failed to classify tweet {tweet.id}; returning fallback analysis")
                    payload = {
                        "sentiment": "neutral",
                        "subnet_id": 0,
                        "subnet_name": "NONE_OF_THE_ABOVE",
                        "content_type": "other",
                        "technical_quality": "none",
                        "market_analysis": "other",
                        "impact_potential": "NONE",
                    }
                    self._cache_set(self._tweet_analysis_cache, cache_key, payload)
                    tweet.analysis = TweetAnalysisBase(
                        sentiment="neutral",
                        subnet_id=0,
                        subnet_name="NONE_OF_THE_ABOVE",
                        content_type="other",
                        technical_quality="none",
                        market_analysis="other",
                        impact_potential="NONE",
                    )
                    self._maybe_log_cache_stats()
                    continue

                # Create analysis object with required fields for validator
                payload = {
                    "sentiment": classification.sentiment.value,
                    "subnet_id": classification.subnet_id,
                    "subnet_name": classification.subnet_name,
                    "content_type": classification.content_type.value,
                    "technical_quality": classification.technical_quality.value,
                    "market_analysis": classification.market_analysis.value,
                    "impact_potential": classification.impact_potential.value,
                }
                self._cache_set(self._tweet_analysis_cache, cache_key, payload)
                tweet.analysis = TweetAnalysisBase(
                    sentiment=classification.sentiment.value,
                    subnet_id=classification.subnet_id,
                    subnet_name=classification.subnet_name,
                    content_type=classification.content_type.value,
                    technical_quality=classification.technical_quality.value,
                    market_analysis=classification.market_analysis.value,
                    impact_potential=classification.impact_potential.value,
                )
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
                cached = self._cache_get(self._telegram_analysis_cache, cache_key)
                if cached is not None:
                    bt.logging.info(
                        f"[Miner][Cache] Telegram cache hit for message_id={msg.id} from validator={validator_hotkey[:12]}.."
                    )
                    msg.analysis = TelegramMessageAnalysis(**cached)
                    self._maybe_log_cache_stats()
                    continue
                
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
                
                # Classify the message group
                classification = None
                try:
                    classification = self.telegram_analyzer.classify_message_group(
                        messages_for_analysis,
                        subnet_id=inherited_subnet_id,
                    )
                except Exception:
                    classification = None

                # If classification fails, fall back to safe defaults to avoid missing analysis.
                if classification is None:
                    bt.logging.warning(f"[Miner] Failed to classify telegram message {msg.id}; returning fallback analysis")
                    from datetime import datetime
                    payload = {
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
                    self._cache_set(self._telegram_analysis_cache, cache_key, payload)
                    msg.analysis = TelegramMessageAnalysis(
                        id=0,  # Placeholder, will be assigned by API
                        message_id=msg.id,
                        sentiment="neutral",
                        subnet_id=int(inherited_subnet_id) if inherited_subnet_id is not None else 0,
                        subnet_name="NONE_OF_THE_ABOVE",
                        content_type="other",
                        technical_quality="none",
                        market_analysis="other",
                        impact_potential="NONE",
                        relevance_confidence=None,
                        analyzed_at=datetime.now().isoformat(),
                    )
                    self._maybe_log_cache_stats()
                    continue
                
                # Create analysis object with required fields for validator
                from datetime import datetime
                payload = {
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
                self._cache_set(self._telegram_analysis_cache, cache_key, payload)
                msg.analysis = TelegramMessageAnalysis(
                    id=0,  # Placeholder, will be assigned by API
                    message_id=msg.id,
                    sentiment=classification.sentiment.value,
                    subnet_id=classification.subnet_id,
                    subnet_name=classification.subnet_name,
                    content_type=classification.content_type.value,
                    technical_quality=classification.technical_quality.value,
                    market_analysis=classification.market_analysis.value,
                    impact_potential=classification.impact_potential.value,
                    relevance_confidence=classification.relevance_confidence,
                    analyzed_at=datetime.now().isoformat(),
                )
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

