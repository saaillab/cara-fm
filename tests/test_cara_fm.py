"""Unit tests for all 9 CARA-FM indices and schema validation."""

from cara_fm import (
    compute_cai,
    compute_cas,
    compute_cei,
    compute_dfs,
    compute_iae,
    compute_ioe,
    compute_kpv,
    compute_mpp,
    compute_yip,
    evaluate_system_card,
    validate_scorecard_dict,
)


def test_all_nine_indices_bounded():
    cei = compute_cei(92.0, model_size_mb=12.0, cpu_latency_ms=22.0)
    dfs_edge = compute_dfs(12.0, peak_ram_gb=0.4, cpu_latency_ms=22.0, offline_capable=True, requires_gpu=False)
    dfs_cloud = compute_dfs(850.0, peak_ram_gb=12.0, cpu_latency_ms=900.0, offline_capable=False, requires_gpu=True)
    assert 0.0 <= cei <= 1.0
    assert dfs_edge == 1.0
    assert dfs_cloud < dfs_edge

    mpp = compute_mpp([0.82, 0.74], high_resource_baseline=0.90)
    assert 0.0 < mpp <= 1.0

    yip = compute_yip(3200.0, 2000.0)
    ioe = compute_ioe(100.0, 72.0)
    iae = compute_iae(0.85, 0.80, 0.75)
    cai = compute_cai(1.0, 0.85, 1.0, 0.70)
    cas = compute_cas(0.76, 0.82)
    kpv_attr = compute_kpv(18, 20, community_attribution_compliant=True)
    kpv_unattr = compute_kpv(18, 20, community_attribution_compliant=False)
    assert yip == 1.0
    assert abs(ioe - 0.28) < 1e-9
    assert 0.0 <= iae <= 1.0 and 0.0 <= cai <= 1.0 and 0.0 <= cas <= 1.0
    assert kpv_attr > kpv_unattr


def test_scorecard_schema_validation():
    card = evaluate_system_card(
        model_id="EfficientNet-B0-Edge",
        task_domain="cassava_and_cloves",
        balanced_accuracy=84.0,
        model_size_mb=20.4,
        peak_ram_gb=0.6,
        cpu_latency_ms=48.0,
        offline_capable=True,
        requires_gpu=False,
        local_language_scores=[0.80, 0.76],
        high_resource_lang_score=0.90,
        local_taxonomy_coverage=1.0,
        swahili_glossary_parity=0.84,
        offline_ui_ready=1.0,
        iks_compatibility=0.72,
    )
    d = card.to_dict()
    assert validate_scorecard_dict(d) == []
    assert d["license"] == "CC-BY-4.0"
    assert d["agricultural_indices"]["YIP"] is None
