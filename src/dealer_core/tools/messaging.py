"""Message channels for send_details. WhatsApp is MOCKED; SMS/email slot in by implementing `MessageChannel`."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Protocol

from ..models import CarView


class MessageChannel(Protocol):
    name: str

    def send(self, to_phone: str, text: str, media_urls: list[str]) -> str:
        """Deliver a message; returns a provider message id."""


@dataclass
class MockChannel:
    """Records what would have been sent. Used for tests and the demo CLI."""

    name: str = "whatsapp"
    sent: list[dict] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def send(self, to_phone: str, text: str, media_urls: list[str]) -> str:
        mid = f"mock-{self.name}-{next(self._ids)}"
        self.sent.append({"id": mid, "to": to_phone, "text": text, "media": list(media_urls)})
        return mid


def compose_details(cars: list[CarView], licence_note: str | None = None) -> tuple[str, list[str]]:
    """Plain-text message with the details we do NOT read aloud (specs, photos, branch). Never includes the floor."""
    lines = []
    media: list[str] = []
    for c in cars:
        lines.append(f"{c.year} {c.make} {c.model} ({c.stock_id}): {c.km:,} km, AED {c.price:,}, {c.color}, {c.regional_spec} spec, "
                     f"{c.fuel}, {c.transmission}. Branch {c.branch_id}.")
        media.extend(c.image_urls[:3])
    if licence_note:
        lines.append(licence_note)
    return "\n".join(lines), media
