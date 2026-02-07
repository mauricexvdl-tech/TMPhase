"""Emergent cancellation analysis.

Tests whether the Phase Cancellation system actually learns destructive
interference, or just acts as two independent feature extractors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evolution.coevolution import PhaseCancellationSystem
from ..evolution.fitness import OracleDataset
from ..neat.network import FeedForwardNetwork
from ..utils.encoding import encode_statement


@dataclass
class CancellationMetrics:
    """Metrics measuring whether real cancellation is happening."""

    # Cosine similarity between E and I vectors
    mean_cos_sim_true: float    # For true statements
    mean_cos_sim_false: float   # For false statements

    # Signal strength (L2 norm of residual)
    mean_residual_true: float
    mean_residual_false: float

    # Cancellation ratio: how much weaker is the residual vs the inputs?
    # ratio < 1 = destructive interference, > 1 = constructive
    cancel_ratio_true: float
    cancel_ratio_false: float

    # Separation: can you tell true from false by signal strength alone?
    strength_separation: float  # (mean_true - mean_false) / (mean_true + mean_false)

    # Anti-alignment score: do I vectors oppose E vectors for false statements?
    # Positive = anti-aligned (real cancellation), negative = aligned
    anti_alignment_true: float
    anti_alignment_false: float


def analyze_cancellation(
    system: PhaseCancellationSystem,
    dataset: OracleDataset,
) -> CancellationMetrics:
    """Analyze whether the system exhibits genuine destructive interference."""
    best_e = system._get_best_or_random(system.embedder_e.population)
    best_i = system._get_best_or_random(system.embedder_i.population)
    best_interference = (
        system._get_best_or_random(system.interference.population)
        if system.interference.population
        else None
    )

    net_e = FeedForwardNetwork(best_e)
    net_i = FeedForwardNetwork(best_i)
    net_int = FeedForwardNetwork(best_interference) if best_interference else None

    cos_sim_true, cos_sim_false = [], []
    residual_true, residual_false = [], []
    cancel_ratio_true, cancel_ratio_false = [], []
    anti_align_true, anti_align_false = [], []

    for item in dataset.items:
        encoded = item.encode(system.config.input_dim)
        pos_e = net_e.activate(encoded)
        pos_i = net_i.activate(encoded)

        # Residual
        if net_int is not None and system.config.use_learned_interference:
            combined = np.concatenate([pos_e, pos_i])
            residual = net_int.activate(combined)
        else:
            residual = pos_e + pos_i

        e_norm = np.linalg.norm(pos_e)
        i_norm = np.linalg.norm(pos_i)
        r_norm = np.linalg.norm(residual)

        # Cosine similarity between E and I
        if e_norm > 1e-8 and i_norm > 1e-8:
            cos = float(np.dot(pos_e, pos_i) / (e_norm * i_norm))
        else:
            cos = 0.0

        # Cancel ratio: residual strength / average input strength
        avg_input = (e_norm + i_norm) / 2.0
        cr = r_norm / (avg_input + 1e-8)

        # Anti-alignment: negative dot product = vectors oppose each other
        anti = -float(np.dot(pos_e, pos_i))

        if item.is_true:
            cos_sim_true.append(cos)
            residual_true.append(r_norm)
            cancel_ratio_true.append(cr)
            anti_align_true.append(anti)
        else:
            cos_sim_false.append(cos)
            residual_false.append(r_norm)
            cancel_ratio_false.append(cr)
            anti_align_false.append(anti)

    mean_res_true = float(np.mean(residual_true)) if residual_true else 0.0
    mean_res_false = float(np.mean(residual_false)) if residual_false else 0.0
    denom = mean_res_true + mean_res_false + 1e-8
    separation = (mean_res_true - mean_res_false) / denom

    return CancellationMetrics(
        mean_cos_sim_true=float(np.mean(cos_sim_true)) if cos_sim_true else 0.0,
        mean_cos_sim_false=float(np.mean(cos_sim_false)) if cos_sim_false else 0.0,
        mean_residual_true=mean_res_true,
        mean_residual_false=mean_res_false,
        cancel_ratio_true=float(np.mean(cancel_ratio_true)) if cancel_ratio_true else 0.0,
        cancel_ratio_false=float(np.mean(cancel_ratio_false)) if cancel_ratio_false else 0.0,
        strength_separation=separation,
        anti_alignment_true=float(np.mean(anti_align_true)) if anti_align_true else 0.0,
        anti_alignment_false=float(np.mean(anti_align_false)) if anti_align_false else 0.0,
    )


def format_analysis(metrics: CancellationMetrics) -> str:
    """Format cancellation analysis as a readable report."""
    lines = [
        "=== Cancellation Analysis ===",
        "",
        "Cosine similarity E<->I vectors:",
        f"  True statements:  {metrics.mean_cos_sim_true:+.4f}",
        f"  False statements: {metrics.mean_cos_sim_false:+.4f}",
        f"  (Negative = opposing directions = cancellation setup)",
        "",
        "Residual signal strength (L2 norm):",
        f"  True statements:  {metrics.mean_residual_true:.4f}",
        f"  False statements: {metrics.mean_residual_false:.4f}",
        f"  Separation:       {metrics.strength_separation:+.4f}",
        f"  (Positive = true signals stronger = cancellation working)",
        "",
        "Cancel ratio (residual / avg input):",
        f"  True statements:  {metrics.cancel_ratio_true:.4f}",
        f"  False statements: {metrics.cancel_ratio_false:.4f}",
        f"  (< 1.0 = destructive interference, > 1.0 = constructive)",
        "",
        "Anti-alignment score (-dot product):",
        f"  True statements:  {metrics.anti_alignment_true:+.4f}",
        f"  False statements: {metrics.anti_alignment_false:+.4f}",
        f"  (Positive = E and I oppose each other)",
    ]

    # Verdict
    lines.append("")
    lines.append("--- Verdict ---")

    genuine = 0
    if metrics.cancel_ratio_false < metrics.cancel_ratio_true:
        lines.append("[+] False statements have MORE cancellation than true ones")
        genuine += 1
    else:
        lines.append("[-] No differential cancellation between true/false")

    if metrics.strength_separation > 0.01:
        lines.append("[+] True residuals are STRONGER than false residuals")
        genuine += 1
    else:
        lines.append("[-] No meaningful signal strength separation")

    if metrics.anti_alignment_false > metrics.anti_alignment_true:
        lines.append("[+] E/I vectors are MORE opposed for false statements")
        genuine += 1
    else:
        lines.append("[-] No differential anti-alignment")

    if genuine >= 2:
        lines.append("=> GENUINE cancellation behavior detected")
    elif genuine == 1:
        lines.append("=> PARTIAL cancellation behavior")
    else:
        lines.append("=> NO cancellation behavior — system acts as simple ensemble")

    return "\n".join(lines)
