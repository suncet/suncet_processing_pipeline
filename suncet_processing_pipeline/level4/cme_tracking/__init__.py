"""Automated CME detection, tracking, and kinematics."""

from .explanation import (
    CMEExplanationArtifacts,
    recompute_evidence_diagnostics,
    write_cme_method_explanation,
)
from .kinematics import KinematicsResult, fit_kinematics

__all__ = [
    "CMEExplanationArtifacts",
    "KinematicsResult",
    "fit_kinematics",
    "recompute_evidence_diagnostics",
    "write_cme_method_explanation",
]
