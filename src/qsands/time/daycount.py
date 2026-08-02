"""
day-count / year-fraction utility 
"""
from __future__ import annotations

from datetime import date

def year_fraction(start:date, end: date, convention: str = "ACT/365") -> float:
    if convention != "ACT/365":
        raise NotImplementedError(f"Convention {convention} not implemented")
    return (end - start).days / 365.0
