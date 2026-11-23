"""Formal nine-index operationalization of the CARA-FM evaluation framework (PMLR Vol. 302)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence
import numpy as np


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_cei(
    balanced_accuracy: float,
    model_size_mb: float,
    cpu_latency_ms: float,
    size_budget_mb: float = 500.0,
    latency_budget_ms: float = 100.0,
) -> float:
    """Compute Computational Efficiency Index (CEI in [0, 1]).

    Combines predictive accuracy with storage compactness relative to the
    CARA-FM <500 MB edge limit and single-image CPU latency budget.
    """
    acc = _clip01(balanced_accuracy / 100.0 if balanced_accuracy > 1.0 else balanced_accuracy)
    size_score = _clip01(1.0 / (1.0 + max(model_size_mb, 0.01) / (0.25 * size_budget_mb)))
    lat_score = _clip01(latency_budget_ms / (latency_budget_ms + max(cpu_latency_ms, 0.1)))
    eff = 0.5 * size_score + 0.5 * lat_score
    return _clip01(0.55 * acc + 0.45 * eff)


def compute_dfs(
    model_size_mb: float,
    peak_ram_gb: float,
    cpu_latency_ms: float,
    offline_capable: bool,
    requires_gpu: bool,
    device_ram_gb: float = 2.0,
    size_cap_mb: float = 500.0,
    latency_cap_ms: float = 250.0,
) -> float:
    """Compute Deployment Feasibility Score (DFS in [0, 1]) against Table 1 constraints.

    Checks offline availability, CPU-only inference, <500 MB artifact budget,
    1-4 GB smartphone RAM headroom, and sub-250 ms latency.
    """
    checks = [
        1.0 if offline_capable else 0.0,
        1.0 if not requires_gpu else 0.0,
        1.0 if model_size_mb <= size_cap_mb else _clip01(size_cap_mb / max(model_size_mb, 1e-3)),
        1.0 if peak_ram_gb <= 0.65 * device_ram_gb else _clip01(device_ram_gb / max(peak_ram_gb, 1e-3)),
        1.0 if cpu_latency_ms <= latency_cap_ms else _clip01(latency_cap_ms / max(cpu_latency_ms, 1e-3)),
    ]
    return float(np.mean(checks))


def compute_mpp(
    local_language_scores: Sequence[float],
    high_resource_baseline: float,
) -> float:
    """Compute Multilingual Performance Parity (MPP in [0, 1]) relative to English/baseline."""
    base = float(high_resource_baseline / 100.0 if high_resource_baseline > 1.0 else high_resource_baseline)
    if base <= 1e-9 or not local_language_scores:
        return 0.0
    ratios = []
    for s in local_language_scores:
        sv = float(s / 100.0 if s > 1.0 else s)
        ratios.append(_clip01(sv / base))
    return float(np.mean(ratios))


def compute_yip(trial_yield_kg_ha: float, baseline_yield_kg_ha: float) -> float:
    """Compute Yield Improvement Potential (YIP in [0, 1]) from controlled field trial data."""
    if baseline_yield_kg_ha <= 0:
        raise ValueError("baseline_yield_kg_ha must be positive")
    rel_gain = (trial_yield_kg_ha - baseline_yield_kg_ha) / baseline_yield_kg_ha
    return _clip01(rel_gain / 0.50)  # 50% yield gain maps to 1.0


def compute_ioe(baseline_input_cost: float, guided_input_cost: float) -> float:
    """Compute Input Optimization Efficiency (IOE in [0, 1]) from cost/chemical reduction."""
    if baseline_input_cost <= 0:
        raise ValueError("baseline_input_cost must be positive")
    saving_frac = (baseline_input_cost - guided_input_cost) / baseline_input_cost
    return _clip01(saving_frac)


def compute_iae(advisory_reach_frac: float, comprehension_rate_frac: float, actionability_frac: float) -> float:
    """Compute Information Access Enhancement (IAE in [0, 1]) across smallholder cohorts."""
    vals = [_clip01(v) for v in (advisory_reach_frac, comprehension_rate_frac, actionability_frac)]
    return float(np.mean(vals))


def compute_cai(
    local_taxonomy_coverage: float,
    swahili_or_local_glossary_parity: float,
    offline_voice_or_simple_ui_ready: float,
    iks_compatibility_score: float,
) -> float:
    """Compute Cultural Alignment Index (CAI in [0, 1]) across local taxonomy and communication readiness."""
    components = [
        _clip01(local_taxonomy_coverage),
        _clip01(swahili_or_local_glossary_parity),
        _clip01(offline_voice_or_simple_ui_ready),
        _clip01(iks_compatibility_score),
    ]
    return float(np.mean(components))


def compute_cas(farmer_adoption_rate: float, trusted_recommendation_rate: float) -> float:
    """Compute Community Acceptance Score (CAS in [0, 1]) from cooperative deployment telemetry."""
    return float(0.5 * (_clip01(farmer_adoption_rate) + _clip01(trusted_recommendation_rate)))


def compute_kpv(
    verified_iks_concepts_retained: int,
    total_target_iks_concepts: int,
    community_attribution_compliant: bool,
) -> float:
    """Compute Knowledge Preservation Value (KPV in [0, 1]) under elder validation."""
    if total_target_iks_concepts <= 0:
        return 0.0
    coverage = _clip01(verified_iks_concepts_retained / float(total_target_iks_concepts))
    return coverage if community_attribution_compliant else 0.5 * coverage


@dataclass
class CARAFMScorecard:
    framework_version: str
    license: str
    model_id: str
    task_domain: str
    hardware_profile: Dict[str, object]
    technical_indices: Dict[str, float]
    agricultural_indices: Dict[str, Optional[float]]
    cultural_indices: Dict[str, Optional[float]]
    governance_metadata: Dict[str, object]
    composite_readiness_score: float
    empirical_status: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_system_card(
    model_id: str,
    task_domain: str,
    balanced_accuracy: float,
    model_size_mb: float,
    peak_ram_gb: float,
    cpu_latency_ms: float,
    offline_capable: bool,
    requires_gpu: bool,
    local_language_scores: Sequence[float],
    high_resource_lang_score: float,
    local_taxonomy_coverage: float,
    swahili_glossary_parity: float,
    offline_ui_ready: float,
    iks_compatibility: float,
    field_trial_metrics: Optional[Dict[str, Optional[float]]] = None,
    supported_languages: Optional[Sequence[str]] = None,
) -> CARAFMScorecard:
    """Evaluate a candidate model against CARA-FM Technical, Agricultural, and Cultural indices."""
    cei = compute_cei(balanced_accuracy, model_size_mb, cpu_latency_ms)
    dfs = compute_dfs(model_size_mb, peak_ram_gb, cpu_latency_ms, offline_capable, requires_gpu)
    mpp = compute_mpp(local_language_scores, high_resource_lang_score)
    cai = compute_cai(local_taxonomy_coverage, swahili_glossary_parity, offline_ui_ready, iks_compatibility)

    ft = field_trial_metrics or {}
    ag = {
        "YIP": ft.get("YIP"),
        "IOE": ft.get("IOE"),
        "IAE": ft.get("IAE"),
    }
    cult = {
        "CAI": round(cai, 4),
        "CAS": ft.get("CAS"),
        "KPV": ft.get("KPV"),
    }
    observed_scores = [cei, dfs, mpp, cai] + [
        float(v) for v in [ag["YIP"], ag["IOE"], ag["IAE"], cult["CAS"], cult["KPV"]] if v is not None
    ]
    composite = float(np.mean(observed_scores))
    status = (
        "full_field_and_technical"
        if all(v is not None for v in [ag["YIP"], ag["IOE"], ag["IAE"], cult["CAS"], cult["KPV"]])
        else "technical_and_cultural_benchmarked_field_trials_pending"
    )

    return CARAFMScorecard(
        framework_version="1.1.0",
        license="CC-BY-4.0",
        model_id=model_id,
        task_domain=task_domain,
        hardware_profile={
            "device_ram_gb": round(peak_ram_gb, 3),
            "model_size_mb": round(model_size_mb, 3),
            "cpu_latency_ms": round(cpu_latency_ms, 2),
            "offline_capable": bool(offline_capable),
            "requires_gpu": bool(requires_gpu),
        },
        technical_indices={
            "CEI": round(cei, 4),
            "DFS": round(dfs, 4),
            "MPP": round(mpp, 4),
        },
        agricultural_indices=ag,
        cultural_indices=cult,
        governance_metadata={
            "community_data_trust": True,
            "farmer_majority_oversight": True,
            "iks_attribution_protocol": True,
            "supported_languages": list(supported_languages or ["en", "sw"]),
        },
        composite_readiness_score=round(composite, 4),
        empirical_status=status,
    )
