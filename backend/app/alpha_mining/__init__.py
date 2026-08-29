"""Independent, extensible Alpha discovery subsystem.

The legacy ``/mining`` implementation deliberately does not import this package.
"""

from app.alpha_mining.contracts import ENGINE_API_VERSION

__all__ = ["ENGINE_API_VERSION"]
