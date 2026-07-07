"""EmotionalContext — maps user.md + soul.md to FAC emotional directives.

Pure logic, no networking. The FAC bridge (plugin or extension) calls
render() to get a conditioned text string before sending to FAC.

FAC receives text via 0x02 frames. We prepend a structured emotion tag:
    [EMO: tone=calm; warmth=0.45; speed=0.9; emphasis=key_points]
    <actual response text>
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# ── Soul baseline (from SOUL.md) ──────────────────────────────────────────
# Senior engineer's assistant: direct, low-warmth, matter-of-fact, no fluff.
SOUL_BASELINE: dict = {
    "tone": "direct",
    "warmth": 0.2,
    "speed": 1.0,
    "emphasis": "none",
}

# ── User state overrides (from USER.md / Sean's profile) ──────────────────
USER_STATE_OVERRIDES: dict[str, dict] = {
    # Frustrated: version mismatch, unclear paths, wasted effort
    "frustrated": {"tone": "calm", "warmth": 0.45, "speed": 0.9, "emphasis": "key_points"},
    # Destructive op pending: serious, clear emphasis on the warning
    "destructive": {"tone": "serious", "warmth": 0.2, "speed": 0.85, "emphasis": "warning"},
    # Action mode: match terse register, speed up
    "action": {"tone": "direct", "warmth": 0.1, "speed": 1.1, "emphasis": "none"},
    # Deep technical ask: neutral, no emotional coloring
    "technical": {"tone": "neutral-professional", "warmth": 0.15, "speed": 1.0, "emphasis": "none"},
}

# Caps — prevent over-conditioning (unnatural speech) or soul violation.
WARMTH_CAP = 0.5
SPEED_MIN = 0.8
SPEED_MAX = 1.2

# Keyword signals for auto-detection
STATE_KEYWORDS: dict[str, list[str]] = {
    "destructive": ["before you delete", "will delete", "destructive", "irreversible", "cannot be undone", "wipe", "rm -rf"],
    "frustrated": ["mismatch", "wrong version", "deployed version", "unclear source", "where is", "can't find", "frustrat"],
    "technical": ["architecture", "protocol", "binary", "websocket", "registry", "schema", "implementation", "deploy"],
    "action": ["run this", "execute", "do it", "now", "apply", "create the", "build the"],
}


@dataclass
class EmotionalState:
    tone: str = SOUL_BASELINE["tone"]
    warmth: float = SOUL_BASELINE["warmth"]
    speed: float = SOUL_BASELINE["speed"]
    emphasis: str = SOUL_BASELINE["emphasis"]

    def merged_with(self, override: Optional[dict]) -> "EmotionalState":
        if not override:
            return self
        return EmotionalState(
            tone=override.get("tone", self.tone),
            warmth=min(override.get("warmth", self.warmth), WARMTH_CAP),
            speed=max(SPEED_MIN, min(override.get("speed", self.speed), SPEED_MAX)),
            emphasis=override.get("emphasis", self.emphasis),
        )


class EmotionalContext:
    """Loads soul/user context and renders FAC-conditioned text."""

    def __init__(
        self,
        soul_path: str = "~/.hermes/SOUL.md",
        user_path: str = "~/.hermes/memory/USER.md",
    ):
        self.soul_path = os.path.expanduser(soul_path)
        self.user_path = os.path.expanduser(user_path)
        self.baseline = self._load_baseline()
        self._user_text = self._read(self.user_path)

    def _read(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    def _load_baseline(self) -> EmotionalState:
        return EmotionalState(
            tone=SOUL_BASELINE["tone"],
            warmth=SOUL_BASELINE["warmth"],
            speed=SOUL_BASELINE["speed"],
            emphasis=SOUL_BASELINE["emphasis"],
        )

    # Detection priority — explicit action/destructive signals win over
    # generic technical keywords that also appear in those phrases.
    DETECT_ORDER = ["destructive", "action", "frustrated", "technical"]

    def detect_state(self, text: str) -> Optional[str]:
        """Keyword-based state detection."""
        lowered = text.lower()
        for state in self.DETECT_ORDER:
            keywords = STATE_KEYWORDS[state]
            if any(kw in lowered for kw in keywords):
                return state
        return None

    def render(self, text: str, state: Optional[str] = None) -> str:
        """Return text prefixed with FAC emotion tag.

        Args:
            text: the actual response text to speak
            state: explicit emotional state ('frustrated', 'destructive',
                   'action', 'technical'). If None, auto-detect from text.
        """
        if state is None:
            state = self.detect_state(text)

        override = USER_STATE_OVERRIDES.get(state) if state else None
        merged = self.baseline.merged_with(override)

        tag = (
            f"[EMO: tone={merged.tone}; warmth={merged.warmth}; "
            f"speed={merged.speed}; emphasis={merged.emphasis}]"
        )
        return f"{tag}\n{text}"
