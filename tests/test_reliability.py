"""
test_reliability.py — Unit tests for reliability controls (Circuit Breaker).

Verifies that:
  1. The Circuit Breaker correctly opens after N failures.
  2. The Circuit Breaker prevents calls while in OPEN state.
  3. The Circuit Breaker attempts recovery (HALF_OPEN) after a cooldown.
"""

import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import CircuitBreaker

class TestCircuitBreaker:
    def test_circuit_breaker_opens_after_threshold(self):
        # Set threshold to 2 for fast testing
        cb = CircuitBreaker(failure_threshold=2, recovery_time_seconds=1)
        
        mock_func = MagicMock(side_effect=ValueError("API Down"))
        
        # Call 1 (Failure)
        with pytest.raises(ValueError):
            cb.call(mock_func)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 1
        
        # Call 2 (Threshold reached)
        with pytest.raises(ValueError):
            cb.call(mock_func)
        assert cb.state == "OPEN"
        assert cb.failure_count == 2

        # Call 3 (Should be blocked immediately without calling mock_func)
        mock_func.reset_mock()
        with pytest.raises(RuntimeError) as exc:
            cb.call(mock_func)
        assert "CircuitBreaker is OPEN" in str(exc.value)
        mock_func.assert_not_called()

    def test_circuit_breaker_recovers_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_time_seconds=0.1)
        
        mock_func = MagicMock(side_effect=ValueError("Fail"))
        
        # Trip the circuit
        with pytest.raises(ValueError):
            cb.call(mock_func)
        assert cb.state == "OPEN"
        
        # Wait for recovery time
        time.sleep(0.2)
        
        # Next call should be HALF_OPEN -> SUCCESS -> CLOSED
        mock_func.side_effect = None
        mock_func.return_value = "Success"
        
        result = cb.call(mock_func)
        assert result == "Success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
