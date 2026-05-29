"""Hydra telemetry color-limit parsing and evaluation."""

from __future__ import annotations

import operator
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUS_COLORS = {
    "green": "#4ec28e",
    "yellow": "#e0a64a",
    "red": "#ef6b6b",
    "unknown": "#5bc4d6",
}


@dataclass(frozen=True)
class RedYellowLimit:
    item: str
    rl: float | None = None
    yl: float | None = None
    yh: float | None = None
    rh: float | None = None

    def evaluate(self, value: float) -> dict[str, Any]:
        if self.rl is not None and value < self.rl:
            return self._status("red", f"value < red low ({self.rl:g})")
        if self.rh is not None and value > self.rh:
            return self._status("red", f"value > red high ({self.rh:g})")
        if self.yl is not None and value < self.yl:
            return self._status("yellow", f"value < yellow low ({self.yl:g})")
        if self.yh is not None and value > self.yh:
            return self._status("yellow", f"value > yellow high ({self.yh:g})")
        return self._status("green", "within green range")

    def _status(self, state: str, reason: str) -> dict[str, Any]:
        return {
            "state": state,
            "color": STATUS_COLORS[state],
            "reason": reason,
            "limits": {
                "rl": self.rl,
                "yl": self.yl,
                "yh": self.yh,
                "rh": self.rh,
            },
        }


@dataclass(frozen=True)
class ValueLimit:
    item: str
    op: str
    value: float
    state: str

    def matches(self, value: float) -> bool:
        func = _OPERATORS.get(self.op)
        if func is None:
            return False
        return bool(func(value, self.value))


class ColorLimitEvaluator:
    def __init__(
        self,
        *,
        red_yellow: dict[str, RedYellowLimit] | None = None,
        values: dict[str, list[ValueLimit]] | None = None,
    ) -> None:
        self.red_yellow = red_yellow or {}
        self.values = values or {}

    @classmethod
    def from_xml(cls, path: str | Path | None) -> "ColorLimitEvaluator":
        if path is None:
            return cls()
        xml_path = Path(path).expanduser()
        if not xml_path.is_file():
            return cls()
        root = ET.parse(xml_path).getroot()
        red_yellow: dict[str, RedYellowLimit] = {}
        values: dict[str, list[ValueLimit]] = {}

        for node in root:
            if not _enabled(node):
                continue
            item = node.attrib.get("item", "").strip()
            if not item:
                continue
            if node.tag == "monitorRedYellow":
                red_yellow[item] = RedYellowLimit(
                    item=item,
                    rl=_float_or_none(node.attrib.get("rl")),
                    yl=_float_or_none(node.attrib.get("yl")),
                    yh=_float_or_none(node.attrib.get("yh")),
                    rh=_float_or_none(node.attrib.get("rh")),
                )
            elif node.tag == "monitorValue":
                state = node.attrib.get("color", "unknown").strip().lower()
                if state not in STATUS_COLORS:
                    state = "unknown"
                expected = _float_or_none(node.attrib.get("value"))
                if expected is None:
                    continue
                values.setdefault(item, []).append(
                    ValueLimit(
                        item=item,
                        op=node.attrib.get("operator", "==").strip(),
                        value=expected,
                        state=state,
                    )
                )
        return cls(red_yellow=red_yellow, values=values)

    def evaluate(self, item: str, value: float) -> dict[str, Any]:
        numeric_limit = self.red_yellow.get(item)
        if numeric_limit is not None:
            return numeric_limit.evaluate(value)

        value_limits = self.values.get(item, ())
        for limit in value_limits:
            if limit.matches(value):
                return {
                    "state": limit.state,
                    "color": STATUS_COLORS[limit.state],
                    "reason": f"value {limit.op} {limit.value:g}",
                    "limits": {"value": limit.value, "operator": limit.op},
                }

        if value_limits:
            return {
                "state": "unknown",
                "color": STATUS_COLORS["unknown"],
                "reason": "no value rule matched",
                "limits": {},
            }
        return {
            "state": "unknown",
            "color": STATUS_COLORS["unknown"],
            "reason": "no color limit configured",
            "limits": {},
        }


def _enabled(node: ET.Element) -> bool:
    return node.attrib.get("enabled", "true").strip().lower() != "false"


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

