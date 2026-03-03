"""
utils.py — Shared utilities for reliability and performance.

Includes:
  - CircuitBreaker: Prevents cascading failures when a downstream service
    (like the LLM API) is experiencing high error rates.
  - Timer: Context manager for precise latency measurement.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")

class CircuitBreaker:
    """
    State machine for preventing cascading failures.
    Transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_time_seconds: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time_seconds
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED" # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute the function if the circuit is not OPEN."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_time:
                logger.info("CircuitBreaker: Attempting HALF_OPEN recovery.")
                self.state = "HALF_OPEN"
            else:
                raise RuntimeError("CircuitBreaker is OPEN. Downstream service is unavailable.")

        try:
            result = func(*args, **kwargs)
            
            # Success logic
            if self.state == "HALF_OPEN":
                logger.info("CircuitBreaker: Recovery successful. Resetting to CLOSED.")
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result

        except Exception as e:
            # Failure logic
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != "OPEN":
                    logger.error(f"CircuitBreaker: Threshold reached ({self.failure_threshold}). Opening circuit for {self.recovery_time}s.")
                self.state = "OPEN"
            
            raise e


class Timer:
    """Context manager for measuring execution time in milliseconds."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000 # ms
        self.interval_sec = self.end - self.start      # s

# Global circuit breaker for LLM
llm_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time_seconds=60)
