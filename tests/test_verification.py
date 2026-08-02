import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from deep_research.config import Config
from deep_research.kb import verification as v
from deep_research.kb.claim_filters import is_review_excluded_claim
from deep_research.models import SearchResult


def test_assessment_search_results_are_rejected_without_blocking_legitimate_exam_reporting():
    assert v._is_assessment_search_result(SearchResult(
        title="Exam 16: Partnership Accounting | Examlex",
        url="https://examlexpp.work/exams/69506-partnership-accounting/2",
        snippet="Multiple choice answers",
    )) is True
    assert v._is_assessment_search_result(SearchResult(
        title="State reports bar exam passage rates fell in 2025",
        url="https://news.example.org/legal/bar-exam-rates",
        snippet="A report on annual passage rates.",
    )) is False


async def test_batch_verification_uses_the_dedicated_verifier_endpoint(monkeypatch):
    config = Config()
    config.kb.extraction_llm_base_url = "http://extractor/v1"
    config.kb.verification_llm_base_url = "http://verifier/v1"
    seen = []

    async def fake_detect(url):
        seen.append(url)
        return "detected-verifier"

    async def fake_verify(_db, _config, claim_id, **kwargs):
        assert kwargs["extraction_model"] == "detected-verifier"
        return SimpleNamespace(status="unverified")

    monkeypatch.setattr(v, "detect_model", fake_detect)
    monkeypatch.setattr(v, "verify_claim", fake_verify)
    outcomes = await v.verify_claims_concurrently(None, config, [{"id": "claim-1"}])
    assert seen == ["http://verifier/v1"]
    assert outcomes[0][1] == "unverified"


# -- _Budget: pure state machine, no I/O -------------------------------------

def test_budget_sources_and_searches_remaining():
    budget = v._Budget(max_sources=2, max_searches=1)
    assert budget.sources_remaining("internal") is True
    assert budget.sources_remaining("external") is True
    assert budget.searches_remaining() is True
    budget.record_source_examined("internal")
    budget.record_source_examined("internal")
    budget.web_searches_used = 1
    assert budget.sources_remaining("internal") is False
    assert budget.searches_remaining() is False


def test_budget_source_phases_are_independent():
    """The bug this guards against: a claim with a few weak internal matches
    burning the whole "sources examined" budget in phase 1 and leaving
    nothing for the web fallback in phase 2. Exhausting one phase's budget
    must not affect the other's."""
    budget = v._Budget(max_sources=1, max_searches=10)
    budget.record_source_examined("internal")
    assert budget.sources_remaining("internal") is False
    assert budget.sources_remaining("external") is True
    budget.record_source_examined("external")
    assert budget.sources_remaining("external") is False
    assert budget.sources_examined == 2


def test_budget_stops_on_any_contradiction():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.contradicts = 1
    assert budget.should_stop() is True


def test_budget_stops_after_two_supports():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports = 2
    assert budget.should_stop() is True
    budget2 = v._Budget(max_sources=10, max_searches=10)
    budget2.supports = 1
    assert budget2.should_stop() is False


def test_budget_stops_after_an_official_weighted_corroboration():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports = 1
    budget.support_weight = 1.0
    assert budget.should_stop() is True


def test_budget_does_not_stop_on_source_budget_exhaustion_alone():
    """Exhausting the *internal* source budget must not itself trigger
    should_stop() -- that's what used to block the web fallback from ever
    being tried. Only a contradiction or 2 supports should stop things;
    running out of sources in a given phase is enforced separately via
    sources_remaining(phase), so the other phase still gets its own budget."""
    budget = v._Budget(max_sources=1, max_searches=10)
    budget.record_source_examined("internal")
    assert budget.should_stop() is False


def test_budget_final_status_mixed_when_both_present():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports, budget.contradicts = 1, 1
    assert budget.final_status() == "mixed"


def test_budget_final_status_contradicted():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.contradicts = 1
    assert budget.final_status() == "contradicted"


def test_budget_final_status_supported():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports = 2
    assert budget.final_status() == "supported"


def test_budget_final_status_unverified_by_default():
    budget = v._Budget(max_sources=10, max_searches=10)
    assert budget.final_status() == "unverified"


# -- _own_evidence_corroboration: pure function, no I/O ----------------------
# Live KB data: a claim ("The Wall Street Crash occurred on October 29th,
# 1929") already backed by history.com, ebsco.com, and explaininghistory.org
# via extraction/merge history sat "unverified" forever, since verify_claim
# only ever counted NEW, separately-discovered claim rows as corroboration --
# never the claim's own pre-existing, already-independent evidence.

def test_own_evidence_corroboration_counts_distinct_reputable_domains():
    sources = [
        {"canonical_uri": "https://history.com/a", "trust_tier_code": None},
        {"canonical_uri": "https://ebsco.com/b", "trust_tier_code": None},
    ]
    count, weight = v._own_evidence_corroboration(sources)
    assert count == 2
    assert weight == 1.0  # two unknown-tier (0.5) sources


def test_own_evidence_corroboration_deduplicates_same_domain():
    # Two different articles from the same outlet are not two independent
    # sources -- history.com twice must count as one.
    sources = [
        {"canonical_uri": "https://history.com/article-a", "trust_tier_code": None},
        {"canonical_uri": "https://history.com/article-b", "trust_tier_code": None},
    ]
    count, weight = v._own_evidence_corroboration(sources)
    assert count == 1
    assert weight == 0.5


def test_own_evidence_corroboration_excludes_social_media():
    sources = [
        {"canonical_uri": "https://www.reddit.com/r/history/comments/x", "trust_tier_code": None},
        {"canonical_uri": "https://history.com/a", "trust_tier_code": None},
    ]
    count, weight = v._own_evidence_corroboration(sources)
    assert count == 1
    assert weight == 0.5


def test_own_evidence_corroboration_uses_trust_tier_weights():
    sources = [
        {"canonical_uri": "https://irs.gov/a", "trust_tier_code": "official"},
    ]
    count, weight = v._own_evidence_corroboration(sources)
    assert count == 1
    assert weight == 1.0


def test_own_evidence_corroboration_empty_for_no_sources():
    assert v._own_evidence_corroboration([]) == (0, 0.0)


async def test_verify_claim_seeds_supports_from_own_evidence(kb_db, monkeypatch):
    """The exact live bug: a claim already backed by two independent
    reputable domains must resolve to "supported" without needing to find
    any new corroborating claim via search."""
    from deep_research.config import load_config

    claim, _ = await kb_db.get_or_create_claim(
        "fact", "The Wall Street Crash occurred on October 29th, 1929.",
    )

    async def fake_own_evidence(claim_id):
        return [
            {"canonical_uri": "https://history.com/articles/1929-stock-market-crash", "trust_tier_code": None},
            {"canonical_uri": "https://ebsco.com/research-starters/history/stock-market-crash-1929", "trust_tier_code": None},
        ]

    monkeypatch.setattr(kb_db, "get_claim_own_evidence_sources", fake_own_evidence)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("should not need external search when own evidence already corroborates")

    monkeypatch.setattr(v, "web_search", fail_if_called)

    result = await v.verify_claim(
        kb_db, load_config(), claim["id"], extraction_model="stub-model",
    )
    assert result.status == "supported"
    assert result.web_searches_used == 0


async def test_verify_claim_max_web_searches_overrides_config(kb_db, monkeypatch):
    """Per-claim search cap override, distinct from run_max_web_searches (the
    shared run-level budget). Confirmed live against a real backlog sweep:
    with the config default of 2, 96% of claims that ended up "unverified"
    had found exactly one supporting source and then hit this per-claim cap
    before a second search could find a corroborating one -- should_stop()
    needs supports >= 2 or support_weight >= 1.0, and a single source below
    "official" trust tier is never enough on its own."""
    from deep_research.config import load_config

    claim, _ = await kb_db.get_or_create_claim("fact", "Some claim needing a real search.")

    captured = {}
    real_budget = v._Budget

    class CapturingBudget(real_budget):
        def __init__(self, max_sources, max_searches):
            captured["max_searches"] = max_searches
            super().__init__(max_sources, max_searches)

    monkeypatch.setattr(v, "_Budget", CapturingBudget)

    async def fake_own_evidence(claim_id):
        return []

    monkeypatch.setattr(kb_db, "get_claim_own_evidence_sources", fake_own_evidence)

    async def fake_web_search(*args, **kwargs):
        return []

    monkeypatch.setattr(v, "web_search", fake_web_search)

    await v.verify_claim(
        kb_db, load_config(), claim["id"], extraction_model="stub-model", max_web_searches=7,
    )
    assert captured["max_searches"] == 7


async def test_verify_claim_defaults_max_web_searches_to_config(kb_db, monkeypatch):
    from deep_research.config import load_config

    claim, _ = await kb_db.get_or_create_claim("fact", "Another claim needing a real search.")

    captured = {}
    real_budget = v._Budget

    class CapturingBudget(real_budget):
        def __init__(self, max_sources, max_searches):
            captured["max_searches"] = max_searches
            super().__init__(max_sources, max_searches)

    monkeypatch.setattr(v, "_Budget", CapturingBudget)

    async def fake_own_evidence(claim_id):
        return []

    monkeypatch.setattr(kb_db, "get_claim_own_evidence_sources", fake_own_evidence)

    async def fake_web_search(*args, **kwargs):
        return []

    monkeypatch.setattr(v, "web_search", fake_web_search)

    config = load_config()
    await v.verify_claim(kb_db, config, claim["id"], extraction_model="stub-model")
    assert captured["max_searches"] == config.kb.verification_max_web_searches


async def test_run_verification_sweep_threads_max_web_searches_to_verify_claim(kb_db, monkeypatch):
    """The per-run override must actually reach verify_claim, not just be
    accepted and dropped -- run_verification_sweep -> verify_claims_concurrently
    -> verify_claim is the real call chain a `verification_sweep` job payload
    goes through (see jobs.py's max_web_searches passthrough)."""
    from deep_research.config import load_config

    captured = {}

    async def fake_verify_claim(kb_db, config, claim_id, **kwargs):
        captured["max_web_searches"] = kwargs.get("max_web_searches")
        return v.VerificationResult(status="unverified", claim_id=claim_id)

    monkeypatch.setattr(v, "verify_claim", fake_verify_claim)

    claim, _ = await kb_db.get_or_create_claim("fact", "A claim eligible for the sweep.", importance_score=0.9)

    await v.run_verification_sweep(
        kb_db, load_config(), trigger="manual", force=True, only_status="unverified", max_web_searches=9,
    )
    assert captured["max_web_searches"] == 9


async def test_run_verification_sweep_threads_diversify_queries_to_verify_claim(kb_db, monkeypatch):
    """Same passthrough requirement as max_web_searches, for the dedicated
    unverified-recheck pass (max_web_searches=3 + diversify_queries=True) --
    see jobs.py's diversify_queries passthrough."""
    from deep_research.config import load_config

    captured = {}

    async def fake_verify_claim(kb_db, config, claim_id, **kwargs):
        captured["diversify_queries"] = kwargs.get("diversify_queries")
        return v.VerificationResult(status="unverified", claim_id=claim_id)

    monkeypatch.setattr(v, "verify_claim", fake_verify_claim)

    claim, _ = await kb_db.get_or_create_claim("fact", "Another claim eligible for the sweep.", importance_score=0.9)

    await v.run_verification_sweep(
        kb_db, load_config(), trigger="manual", force=True, only_status="unverified", diversify_queries=True,
    )
    assert captured["diversify_queries"] is True


# -- eligibility / check_status: settled vs. inconclusive claims -------------
# A settled verdict (supported/contradicted/mixed) is never auto-rechecked.
# An "unverified" (inconclusive) first pass is a different case -- it gets a
# second look automatically once UNVERIFIED_RETRY_COOLDOWN_HOURS has passed,
# rather than being abandoned forever the moment verification_attempted_at is
# set. claim_check_status must always agree with is_claim_eligible_for_verification
# (see the docstring on claim_check_status) -- these tests check both in step.

def _claim(**overrides) -> dict:
    base = {
        "status": "unverified",
        "importance_score": 0.9,
        "verification_attempted_at": None,
        "verification_override": None,
    }
    base.update(overrides)
    return base


def test_never_attempted_claim_above_threshold_is_eligible():
    claim = _claim()
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is True
    assert v.claim_check_status(claim, threshold=0.8) == "auto_check"


def test_never_attempted_claim_below_threshold_is_not_eligible():
    claim = _claim(importance_score=0.5)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "auto_skip"


def test_settled_claim_never_auto_rechecked_even_long_after_attempt():
    old = datetime.now(timezone.utc) - timedelta(days=365)
    claim = _claim(status="supported", verification_attempted_at=old)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "checked"


def test_contradicted_and_mixed_claims_are_also_settled():
    old = datetime.now(timezone.utc) - timedelta(days=365)
    for status in ("contradicted", "mixed"):
        claim = _claim(status=status, verification_attempted_at=old)
        assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
        assert v.claim_check_status(claim, threshold=0.8) == "checked"


def test_unverified_claim_within_cooldown_is_not_yet_eligible():
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    claim = _claim(status="unverified", verification_attempted_at=recent)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "checked_pending_retry"


def test_unverified_claim_past_cooldown_is_eligible_again():
    old = datetime.now(timezone.utc) - timedelta(hours=v.UNVERIFIED_RETRY_COOLDOWN_HOURS + 1)
    claim = _claim(status="unverified", verification_attempted_at=old)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is True
    assert v.claim_check_status(claim, threshold=0.8) == "auto_check"


def test_unverified_claim_past_cooldown_but_below_threshold_is_auto_skip():
    old = datetime.now(timezone.utc) - timedelta(hours=v.UNVERIFIED_RETRY_COOLDOWN_HOURS + 1)
    claim = _claim(status="unverified", verification_attempted_at=old, importance_score=0.5)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "auto_skip"


def test_manual_exclude_always_wins_regardless_of_attempt_state():
    claim = _claim(status="supported", verification_attempted_at=datetime.now(timezone.utc), verification_override="exclude")
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "manual_exclude"


def test_force_bypasses_the_attempted_and_cooldown_gate():
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    claim = _claim(status="unverified", verification_attempted_at=recent)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8, force=True) is True


def test_deprecated_claim_is_never_eligible_even_if_never_attempted():
    # The losing side of a claim merge -- just a pointer to the real claim
    # now, not a live fact worth a verification pass, regardless of
    # importance score or whether it was ever individually checked.
    claim = _claim(status="deprecated", verification_attempted_at=None, importance_score=0.99)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "deprecated"


def test_deprecated_claim_is_never_eligible_even_with_force():
    claim = _claim(status="deprecated", verification_attempted_at=None, importance_score=0.99)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8, force=True) is False


# -- only_status: a targeted backlog sweep must not reopen settled claims ----

def test_only_status_excludes_a_claim_in_a_different_status():
    claim = _claim(status="supported", verification_attempted_at=None)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.0, only_status="unverified") is False


def test_only_status_includes_a_matching_never_attempted_claim():
    claim = _claim(status="unverified", verification_attempted_at=None)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.0, only_status="unverified") is True


def test_only_status_with_force_reopens_a_matching_previously_inconclusive_claim():
    old = datetime.now(timezone.utc) - timedelta(hours=1)  # still within the retry cooldown
    claim = _claim(status="unverified", verification_attempted_at=old)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.0, only_status="unverified") is False
    assert v.is_claim_eligible_for_verification(claim, threshold=0.0, force=True, only_status="unverified") is True


def test_only_status_does_not_reopen_a_settled_claim_even_with_force():
    # The whole point of only_status: force=True alone would otherwise also
    # reopen already-settled supported/contradicted claims across the KB.
    old = datetime.now(timezone.utc) - timedelta(days=365)
    claim = _claim(status="supported", verification_attempted_at=old)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.0, force=True) is True
    assert v.is_claim_eligible_for_verification(claim, threshold=0.0, force=True, only_status="unverified") is False


# -- _classify_relationship: verification_context steers the comparison -----

async def test_classify_relationship_includes_context_when_given():
    seen = {}

    class FakeLLM:
        async def chat(self, messages):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"message": {"content": '{"relationship": "supports", "confidence": 0.9, "reasoning": "ok"}'}}]}

    await v._classify_relationship(
        FakeLLM(), "Industrial buildings use more electricity than residential.",
        "Datacenters use far more electricity per square foot than industrial buildings.",
        context="compare specifically against datacenter usage",
    )

    assert "compare specifically against datacenter usage" in seen["content"]


async def test_classify_relationship_omits_context_line_when_absent():
    seen = {}

    class FakeLLM:
        async def chat(self, messages):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"message": {"content": '{"relationship": "supports", "confidence": 0.9, "reasoning": "ok"}'}}]}

    await v._classify_relationship(FakeLLM(), "Claim A text.", "Claim B text.")

    assert "Additional context" not in seen["content"]


async def test_classify_relationship_does_not_turn_malformed_number_into_contradiction():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": json.dumps({
                "relationship": "contradicts",
                "confidence": 0.95,
                "reasoning": "5,48 must mean 5,480, which differs from 5,048.62.",
            })}}]}

    result = await v._classify_relationship(
        FakeLLM(),
        "The NASDAQ peaked at 5,48 on March 10th, 2000.",
        "The Nasdaq reached 5,048.62 in 2000.",
    )

    assert result == {
        "relationship": "unrelated",
        "confidence": 0.0,
        "reasoning": "A malformed numeric grouping cannot establish a contradiction.",
    }


async def test_classify_relationship_preserves_real_well_formed_numeric_contradiction():
    class FakeLLM:
        async def chat(self, messages):
            if "final gate" in messages[0]["content"]:
                return {"choices": [{"message": {"content": json.dumps({
                    "confirmed": True,
                    "confidence": 0.98,
                    "reasoning": "Both claims give incompatible values for the same peak on the same date.",
                })}}]}
            return {"choices": [{"message": {"content": json.dumps({
                "relationship": "contradicts",
                "confidence": 0.95,
                "reasoning": "The two well-formed values differ.",
            })}}]}

    result = await v._classify_relationship(
        FakeLLM(),
        "The NASDAQ peaked at 5,480 on March 10th, 2000.",
        "The Nasdaq reached 5,048.62 in 2000.",
    )

    assert result["relationship"] == "contradicts"


async def test_classify_relationship_rejects_false_chronological_contradiction():
    calls = 0

    class FakeLLM:
        async def chat(self, messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"choices": [{"message": {"content": json.dumps({
                    "relationship": "contradicts",
                    "confidence": 0.95,
                    "reasoning": "The first pass incorrectly reversed the chronology.",
                })}}]}
            return {"choices": [{"message": {"content": json.dumps({
                "confirmed": False,
                "confidence": 0.99,
                "reasoning": "A rate change in 1916 is compatible with no tax before 1913.",
            })}}]}

    result = await v._classify_relationship(
        FakeLLM(),
        "The United States had no income tax until 1913.",
        "The top income tax rate was raised to 15% in 1916.",
    )

    assert calls == 2
    assert result == {
        "relationship": "unrelated",
        "confidence": 0.99,
        "reasoning": "A rate change in 1916 is compatible with no tax before 1913.",
    }


async def test_classify_relationship_requires_confident_second_pass():
    class FakeLLM:
        async def chat(self, messages):
            if "final gate" in messages[0]["content"]:
                return {"choices": [{"message": {"content": json.dumps({
                    "confirmed": True,
                    "confidence": 0.4,
                    "reasoning": "The time scope is unclear.",
                })}}]}
            return {"choices": [{"message": {"content": json.dumps({
                "relationship": "contradicts",
                "confidence": 0.95,
                "reasoning": "The values differ.",
            })}}]}

    result = await v._classify_relationship(
        FakeLLM(), "Revenue was $5 million.", "Revenue was $8 million."
    )

    assert result["relationship"] == "unrelated"
    assert result["confidence"] == 0.4


async def test_classify_relationship_never_links_subjectless_cost_claims():
    class FakeLLM:
        async def chat(self, messages):
            raise AssertionError("subjectless claims should be rejected before the model call")

    result = await v._classify_relationship(
        FakeLLM(),
        "The total cost had risen to 23 million.",
        "It cost $1.4 billion",
    )

    assert result == {
        "relationship": "unrelated",
        "confidence": 1.0,
        "reasoning": "At least one measurement claim has no identifiable subject.",
    }


async def test_classify_relationship_never_links_unresolved_personal_pronouns():
    class FakeLLM:
        async def chat(self, messages):
            raise AssertionError("unresolved pronouns should be rejected before the model call")

    result = await v._classify_relationship(
        FakeLLM(),
        "He retired with a $417 million severance package.",
        "The largest severance package is over one billion dollars.",
    )

    assert result["relationship"] == "unrelated"
    assert result["confidence"] == 1.0


def test_disjoint_decades_are_separate_time_scopes():
    assert v._has_disjoint_time_scopes(
        "Households in the 1980s were significantly smaller and accommodated more people.",
        "Households in the 1950s and 1960s were much larger.",
    ) is True
    assert v._has_disjoint_time_scopes(
        "Households in the 1960s were large.",
        "Households in the 1960s often included two parents and children.",
    ) is False


def test_explicit_cross_period_comparison_is_not_short_circuited():
    assert v._has_disjoint_time_scopes(
        "Households became smaller from the 1960s to the 1980s.",
        "Households in the 1950s were larger than in the 1980s.",
    ) is False


async def test_classify_relationship_skips_disjoint_decades_before_model_call():
    class FakeLLM:
        async def chat(self, messages):
            raise AssertionError("disjoint periods should be rejected before the model call")

    result = await v._classify_relationship(
        FakeLLM(),
        "Households in the 1980s were significantly smaller and accommodated more people on average.",
        "Households in the 1950s and 1960s were much larger, often comprised of two parents with children.",
    )

    assert result == {
        "relationship": "unrelated",
        "confidence": 1.0,
        "reasoning": "The claims describe separate, non-overlapping time periods.",
    }


def test_dated_nobody_wants_to_work_refrain_is_review_excluded():
    for text in (
        "In 1952, nobody wanted to work.",
        "In 1979, no one wanted to work anymore.",
        "In 2014, nobody wants to work.",
    ):
        assert is_review_excluded_claim(text) is True

    assert is_review_excluded_claim(
        "In 2022, a survey found that one in five executives said no one wants to work anymore."
    ) is False


async def test_review_excluded_refrain_skips_relationship_model_call():
    class FakeLLM:
        async def chat(self, messages):
            raise AssertionError("review-excluded claims should not reach the model")

    result = await v._classify_relationship(
        FakeLLM(),
        "In 1952, nobody wanted to work.",
        "Job opportunities were plentiful but workers had to fight to get hired.",
    )

    assert result["relationship"] == "unrelated"
    assert result["confidence"] == 1.0


def test_entity_overlap_rejects_parallel_claims_about_different_empires():
    roman = [
        {"name": "Roman Empire", "type": "organization"},
        {"name": "Emperor Trajan", "type": "person"},
        {"name": "Empire", "type": "concept"},
    ]
    mughal = [{"name": "Mughal Empire", "type": "organization"}]

    assert v._entity_mentions_overlap(roman, mughal) is False


def test_entity_overlap_allows_exact_and_expanded_name_matches():
    assert v._entity_mentions_overlap(
        [{"name": "Roman Empire", "type": "organization"}],
        [{"name": "Roman Empire", "type": "concept"}],
    ) is True
    assert v._entity_mentions_overlap(
        [{"name": "Trajan", "type": "person"}],
        [{"name": "Emperor Trajan", "type": "person"}],
    ) is True


def test_missing_entity_metadata_preserves_model_fallback():
    assert v._entity_mentions_overlap([], [{"name": "Roman Empire", "type": "organization"}]) is True


# -- _suggest_search_query: repeating a failed query wastes the retry --------

async def test_suggest_search_query_uses_llm_suggestion():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"query": "specific alternate query"}'}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", ["Some claim text."])
    assert result == "specific alternate query"


async def test_suggest_search_query_falls_back_to_claim_text_on_garbage_response():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "not json"}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", ["Some claim text."])
    assert result == "Some claim text."


async def test_suggest_search_query_falls_back_to_claim_text_on_empty_query():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"query": ""}'}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", ["Some claim text."])
    assert result == "Some claim text."


async def test_suggest_search_query_falls_back_to_claim_text_on_placeholder_bracket():
    """Live KB data: when a claim never names the specific company/deal/date
    and 'don't repeat prior queries' leaves the model nothing left to vary,
    it can fabricate a fill-in-the-blank query like "...the deal in [specific
    event or time period]" instead of admitting there's no more detail
    available. A query containing a literal bracket would search for that
    bracket text, not real content, so it must be rejected the same as an
    empty or unparseable suggestion."""
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {
                "content": '{"query": "Goldman Sachs risk assessment and advocacy for the deal in [specific event or time period]"}',
            }}]}

    result = await v._suggest_search_query(FakeLLM(), "Goldman Sachs pushed for the deal.", ["a prior query"])
    assert result == "Goldman Sachs pushed for the deal."


async def test_suggest_search_query_falls_back_to_claim_text_on_vague_placeholder_phrase():
    """Live KB data, one step further: told not to invent a bracketed
    placeholder like "[time period]", the model can just as readily write
    the identical fabrication in plain English instead -- "What was the
    average wage for workers during this time period?" for a claim that
    never mentions any time period at all. Same non-content, no brackets, so
    it must be rejected the same way."""
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {
                "content": '{"query": "What was the average wage for workers during this time period?"}',
            }}]}

    result = await v._suggest_search_query(FakeLLM(), "Workers were paid pretty well.", ["a prior query"])
    assert result == "Workers were paid pretty well."


@pytest.mark.parametrize("phrase", [
    "average wage during this time period",
    "average wage during that time period",
    "prices in this era",
    "prices during that era",
    "wages at this point in time",
    "wages during this time",
    "wages at that time",
])
def test_is_fabricated_placeholder_query_catches_vague_phrases_without_brackets(phrase):
    assert v._is_fabricated_placeholder_query(phrase) is True


def test_is_fabricated_placeholder_query_false_for_genuine_queries():
    assert v._is_fabricated_placeholder_query("Japan asset bubble household savings 1980s") is False
    assert v._is_fabricated_placeholder_query("average wage in Japan in the 1980s") is False


def test_only_meaningfully_different_generated_query_is_alternate():
    claim = "The Nasdaq peaked in March 2000."

    assert v._is_alternate_search_query(claim, "NASDAQ March 2000 historical peak") is True
    assert v._is_alternate_search_query(claim, "  the NASDAQ peaked in March 2000. ") is False


# -- _is_duplicate_query: live regression, the model repeating itself -------
# Observed live: SEARCH_QUERY_SUGGESTION_PROMPT explicitly tells the model
# not to repeat a prior query, but a smaller local model sometimes does
# anyway -- one claim's retry history showed the same exact query verbatim
# 5 times, each repeat wasting a real search-budget slot on results already
# seen instead of exploring a genuinely new angle.

def test_is_duplicate_query_matches_case_and_whitespace_insensitively():
    tried = ["Energy industry growth projections compared to technology and healthcare sectors by 2030"]
    assert v._is_duplicate_query(
        "  ENERGY industry growth projections compared to technology and healthcare sectors by 2030  ", tried,
    ) is True


def test_is_duplicate_query_false_for_a_genuinely_new_query():
    tried = ["Energy industry growth projections compared to technology and healthcare sectors by 2030"]
    assert v._is_duplicate_query("Global energy demand forecast 2030 IEA report", tried) is False


def test_is_duplicate_query_false_for_empty_history():
    assert v._is_duplicate_query("Any query at all", []) is False


# -- _suggest_diverse_search_queries: batch query generation ----------------
# Built for a dedicated "recheck unverified claims" pass, not yet wired into
# the routine backlog sweep -- see run_verification_sweep's diversify_queries
# docstring.

async def test_suggest_diverse_search_queries_returns_llm_suggestions():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {
                "content": '{"queries": ["angle one query", "angle two query"]}',
            }}]}

    result = await v._suggest_diverse_search_queries(FakeLLM(), "Some claim text.", [], n=3)
    assert result == ["angle one query", "angle two query"]


async def test_suggest_diverse_search_queries_caps_to_n():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"queries": ["a", "b", "c", "d", "e"]}'}}]}

    result = await v._suggest_diverse_search_queries(FakeLLM(), "Some claim text.", [], n=2)
    assert result == ["a", "b"]


async def test_suggest_diverse_search_queries_returns_empty_on_garbage_response():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "not json"}}]}

    result = await v._suggest_diverse_search_queries(FakeLLM(), "Some claim text.", [])
    assert result == []


async def test_suggest_diverse_search_queries_filters_placeholder_brackets_and_blanks():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": json.dumps({
                "queries": [
                    "a genuine query",
                    "",
                    "the deal in [specific event or time period]",
                    "   ",
                    "wages during this time period",
                    "another genuine query",
                ],
            })}}]}

    result = await v._suggest_diverse_search_queries(FakeLLM(), "Some claim text.", [], n=10)
    assert result == ["a genuine query", "another genuine query"]


# -- _dedupe_queries_by_similarity: near-duplicate rewording detection ------
# Observed live: raising the per-claim search cap mostly bought near-duplicate
# rewordings ("...official statistics" / "...official data" / "...verified
# data") of the same failed search, not genuinely new angles -- exact-string
# dedup alone doesn't catch this since the wording differs each time.

async def test_dedupe_queries_by_similarity_drops_near_duplicate_of_tried_query(monkeypatch):
    vectors = {
        "tried query": [1.0, 0.0],
        "near-duplicate rewording": [0.99, 0.1411],  # cosine ~0.99 to "tried query"
        "genuinely different angle": [0.0, 1.0],
    }

    async def fake_embed_texts(texts, base_url, model):
        return [vectors[t] for t in texts]

    monkeypatch.setattr(v, "embed_texts", fake_embed_texts)

    result = await v._dedupe_queries_by_similarity(
        ["near-duplicate rewording", "genuinely different angle"],
        ["tried query"], "http://fake", "fake-model",
    )
    assert result == ["genuinely different angle"]


async def test_dedupe_queries_by_similarity_drops_near_duplicates_within_the_batch(monkeypatch):
    vectors = {
        "first phrasing": [1.0, 0.0],
        "second phrasing of the same thing": [0.99, 0.1411],
        "actually different angle": [0.0, 1.0],
    }

    async def fake_embed_texts(texts, base_url, model):
        return [vectors[t] for t in texts]

    monkeypatch.setattr(v, "embed_texts", fake_embed_texts)

    result = await v._dedupe_queries_by_similarity(
        ["first phrasing", "second phrasing of the same thing", "actually different angle"],
        [], "http://fake", "fake-model",
    )
    assert result == ["first phrasing", "actually different angle"]


async def test_dedupe_queries_by_similarity_drops_exact_duplicates_without_embedding(monkeypatch):
    async def fail_if_called(texts, base_url, model):
        raise AssertionError("should not need embeddings once every candidate is an exact duplicate")

    monkeypatch.setattr(v, "embed_texts", fail_if_called)

    result = await v._dedupe_queries_by_similarity(
        ["Tried Query"], ["tried query"], "http://fake", "fake-model",
    )
    assert result == []


async def test_dedupe_queries_by_similarity_falls_back_to_exact_match_only_on_embed_failure(monkeypatch):
    async def failing_embed_texts(texts, base_url, model):
        raise RuntimeError("embedding backend unreachable")

    monkeypatch.setattr(v, "embed_texts", failing_embed_texts)

    result = await v._dedupe_queries_by_similarity(
        ["candidate one", "candidate two"], ["tried query"], "http://fake", "fake-model",
    )
    assert result == ["candidate one", "candidate two"]


async def test_verify_claim_diversify_queries_drains_batch_across_iterations(kb_db, monkeypatch):
    """diversify_queries=True should call the batch suggester once (not once
    per search, unlike the default single-query path) and drain its results
    one at a time across loop iterations, only asking for a fresh batch once
    that one is exhausted. The very first search always uses the literal
    claim text regardless of this flag, same as the default path -- the
    batch suggester only kicks in from the second search onward, once
    tried_queries is non-empty."""
    from dataclasses import dataclass

    from deep_research.config import load_config
    from deep_research.models import SearchResult

    claim, _ = await kb_db.get_or_create_claim(
        "fact", "A distinctive test claim about zorbnaxian trade tariffs in 1994.",
    )
    config = load_config()
    config.kb.verification_max_web_searches = 3

    search_calls: list[str] = []

    async def fake_web_search(query, cfg, **kwargs):
        search_calls.append(query)
        return [SearchResult(title="Irrelevant", url=f"https://example.test/{len(search_calls)}", snippet="")]

    suggest_diverse_calls = 0

    async def fake_suggest_diverse(llm, claim_text, tried_queries, context=None, n=3):
        nonlocal suggest_diverse_calls
        suggest_diverse_calls += 1
        return ["diverse query one", "diverse query two"]

    async def fake_dedupe(candidates, tried_queries, base_url, model, threshold=0.92):
        return list(candidates)

    async def fail_if_called_single(llm, claim_text, tried_queries, context=None):
        raise AssertionError("should not fall back to the single-query suggester while the batch has queries left")

    @dataclass
    class _FailedIngest:
        status: str = "failed"
        source_id: str | None = None

    async def fake_ingest_web_page(*args, **kwargs):
        return _FailedIngest()

    monkeypatch.setattr(v, "web_search", fake_web_search)
    monkeypatch.setattr(v, "_suggest_diverse_search_queries", fake_suggest_diverse)
    monkeypatch.setattr(v, "_dedupe_queries_by_similarity", fake_dedupe)
    monkeypatch.setattr(v, "_suggest_search_query", fail_if_called_single)
    monkeypatch.setattr(v, "ingest_web_page", fake_ingest_web_page)

    await v.verify_claim(
        kb_db, config, claim["id"], extraction_model="stub-model", diversify_queries=True,
    )

    assert suggest_diverse_calls == 1
    assert search_calls == [claim["canonical_text"], "diverse query one", "diverse query two"]


async def test_verify_claim_skips_web_search_for_a_repeated_query(kb_db, monkeypatch):
    """Integration-level version of the same live bug: verify_claim's
    external-search loop must not actually call web_search() again once the
    suggested query duplicates one already tried, even though nothing
    prevents the model itself from suggesting one."""
    from dataclasses import dataclass

    from deep_research.config import load_config
    from deep_research.models import SearchResult

    claim, _ = await kb_db.get_or_create_claim(
        "fact", "A distinctive test claim about zorbnaxian trade tariffs in 1994.",
    )
    config = load_config()
    config.kb.verification_max_web_searches = 4

    search_calls = []

    async def fake_web_search(query, cfg, **kwargs):
        search_calls.append(query)
        # Non-empty (verify_claim's loop breaks outright on an empty
        # result), but resolved as a failed ingest below so nothing further
        # is actually fetched over the network.
        return [SearchResult(title="Irrelevant", url="https://example.test/irrelevant", snippet="")]

    async def fake_suggest_query(llm, claim_text, tried_queries, context=None):
        # Simulate the model ignoring "don't repeat" and echoing the first
        # query (the raw claim text) back verbatim every time.
        return claim_text

    @dataclass
    class _FailedIngest:
        status: str = "failed"
        source_id: str | None = None

    async def fake_ingest_web_page(*args, **kwargs):
        return _FailedIngest()

    monkeypatch.setattr(v, "web_search", fake_web_search)
    monkeypatch.setattr(v, "_suggest_search_query", fake_suggest_query)
    monkeypatch.setattr(v, "ingest_web_page", fake_ingest_web_page)

    result = await v.verify_claim(
        kb_db, config, claim["id"], extraction_model="stub-model",
    )

    # Exactly one real search (the first, raw-claim-text attempt) -- every
    # later loop iteration recognized the suggestion as a repeat and skipped
    # the wasted call, even though the budget allowed up to 4 attempts.
    assert search_calls == [claim["canonical_text"]]
    assert result.web_searches_used == 4


async def test_suggest_search_query_includes_context_in_the_prompt():
    seen = {}

    class FakeLLM:
        async def chat(self, messages):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"message": {"content": '{"query": "datacenter electricity usage vs industrial"}'}}]}

    result = await v._suggest_search_query(
        FakeLLM(), "Industrial buildings use more electricity than residential.", [],
        context="compare specifically against datacenter usage",
    )

    assert result == "datacenter electricity usage vs industrial"
    assert "compare specifically against datacenter usage" in seen["content"]


async def test_suggest_search_query_uses_llm_even_on_first_attempt_when_context_given():
    # Normally the very first search just uses the raw claim text (see
    # verify_claim) -- this only tests that _suggest_search_query itself
    # produces a sensible result when called with no tried_queries yet but
    # context present, which verify_claim does specifically to cover that case.
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"query": "context-aware query"}'}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", [], context="a specific angle")
    assert result == "context-aware query"


# -- _examine_candidates resilience (hardening pass) -------------------------
# A transient failure classifying one candidate must not abort examination of
# the rest, and must not propagate out of _examine_candidates at all -- this
# is the exact gap the hardening pass fixed (previously this would raise all
# the way out of verify_claim, discarding every support/contradiction found
# earlier in the same run).

async def test_examine_candidates_survives_one_failing_comparison(kb_db, monkeypatch):
    target, _ = await kb_db.get_or_create_claim("fact", "Target claim for resilience test.")
    other_a, _ = await kb_db.get_or_create_claim("fact", "Other claim A.")
    other_b, _ = await kb_db.get_or_create_claim("fact", "Other claim B.")
    source_a, _ = await kb_db.get_or_create_source(source_type_code="web", canonical_uri="http://a.example", canonical_key="a")
    source_b, _ = await kb_db.get_or_create_source(source_type_code="web", canonical_uri="http://b.example", canonical_key="b")
    version_a, _ = await kb_db.add_source_version(source_a["id"], content_hash="h1", snapshot_path="/tmp/a", http_status=200, mime_type="text/html")
    version_b, _ = await kb_db.add_source_version(source_b["id"], content_hash="h2", snapshot_path="/tmp/b", http_status=200, mime_type="text/html")
    artifact_a, _ = await kb_db.upsert_artifact(artifact_id="art-a", source_version_id=version_a["id"], artifact_type="clean_text", storage_path="/tmp/a.txt", content_hash="h1", chunk_params_hash="p1")
    artifact_b, _ = await kb_db.upsert_artifact(artifact_id="art-b", source_version_id=version_b["id"], artifact_type="clean_text", storage_path="/tmp/b.txt", content_hash="h2", chunk_params_hash="p1")
    chunk_a = await kb_db.add_chunk(artifact_a["id"], 0, "chunk a", "chash-a")
    chunk_b = await kb_db.add_chunk(artifact_b["id"], 0, "chunk b", "chash-b")
    await kb_db.add_claim_evidence(claim_id=other_a["id"], artifact_chunk_id=chunk_a["id"], source_id=source_a["id"], source_version_id=version_a["id"])
    await kb_db.add_claim_evidence(claim_id=other_b["id"], artifact_chunk_id=chunk_b["id"], source_id=source_b["id"], source_version_id=version_b["id"])

    call_count = {"n": 0}

    async def flaky_classify(llm, a, b, context=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("simulated transient LLM failure")
        return {"relationship": "unrelated", "confidence": 0.1, "reasoning": "test"}

    monkeypatch.setattr(v, "_classify_relationship", flaky_classify)

    budget = v._Budget(max_sources=10, max_searches=1)
    examined_source_ids = set()
    contradiction_ids = []
    ranked = [(other_a, 0.9), (other_b, 0.8)]

    await v._examine_candidates(kb_db, None, None, target, ranked, budget, examined_source_ids, contradiction_ids)

    assert call_count["n"] == 2  # both candidates were attempted despite the first raising
    assert budget.sources_examined == 2  # progress wasn't lost after the failure


# -- social-media-only sources can't settle a verification -------------------
# Reddit/Instagram/Facebook are unvetted user-generated content -- fine to
# read, but if that's the *only* evidence a candidate claim has, it must
# never single-handedly mark the target claim supported/contradicted.

async def test_examine_candidates_skips_llm_call_for_social_media_only_source(kb_db, monkeypatch):
    target, _ = await kb_db.get_or_create_claim("fact", "Target claim for social-media test.")
    reddit_claim, _ = await kb_db.get_or_create_claim("fact", "A claim only backed by a Reddit post.")
    real_claim, _ = await kb_db.get_or_create_claim("fact", "A claim backed by a real news source.")

    reddit_source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://www.reddit.com/r/test/comments/abc", canonical_key="reddit-abc",
    )
    real_source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://www.example-news.example/article", canonical_key="real-article",
    )
    reddit_version, _ = await kb_db.add_source_version(reddit_source["id"], content_hash="h1", snapshot_path="/tmp/r", http_status=200, mime_type="text/html")
    real_version, _ = await kb_db.add_source_version(real_source["id"], content_hash="h2", snapshot_path="/tmp/n", http_status=200, mime_type="text/html")
    reddit_artifact, _ = await kb_db.upsert_artifact(artifact_id="art-reddit", source_version_id=reddit_version["id"], artifact_type="clean_text", storage_path="/tmp/r.txt", content_hash="h1", chunk_params_hash="p1")
    real_artifact, _ = await kb_db.upsert_artifact(artifact_id="art-real", source_version_id=real_version["id"], artifact_type="clean_text", storage_path="/tmp/n.txt", content_hash="h2", chunk_params_hash="p1")
    reddit_chunk = await kb_db.add_chunk(reddit_artifact["id"], 0, "reddit chunk", "chash-r")
    real_chunk = await kb_db.add_chunk(real_artifact["id"], 0, "real chunk", "chash-n")
    await kb_db.add_claim_evidence(claim_id=reddit_claim["id"], artifact_chunk_id=reddit_chunk["id"], source_id=reddit_source["id"], source_version_id=reddit_version["id"])
    await kb_db.add_claim_evidence(claim_id=real_claim["id"], artifact_chunk_id=real_chunk["id"], source_id=real_source["id"], source_version_id=real_version["id"])

    classified = []

    async def fake_classify(llm, a, b, context=None):
        classified.append(b)
        return {"relationship": "supports", "confidence": 0.9, "reasoning": "test"}

    monkeypatch.setattr(v, "_classify_relationship", fake_classify)

    budget = v._Budget(max_sources=10, max_searches=1)
    ranked = [(reddit_claim, 0.9), (real_claim, 0.8)]

    await v._examine_candidates(kb_db, None, None, target, ranked, budget, set(), [], supporting_ids=[])

    assert real_claim["canonical_text"] in classified
    assert reddit_claim["canonical_text"] not in classified  # never sent to the LLM at all
    assert budget.supports == 1  # only the real source's support counted
    assert budget.sources_examined == 2  # the reddit source still cost budget -- it was looked at


async def test_rank_candidates_degrades_gracefully_when_embedding_backend_down(kb_db, monkeypatch):
    """Real bug found while verifying the web UI's verify-claim route with
    Ollama stopped: an unreachable embedding backend raised all the way out
    of verify_claim as an uncaught httpx.ConnectError, discarding the whole
    verification attempt. Every other embed_texts call site in the codebase
    is best-effort; this one should be too."""
    from deep_research.config import load_config

    claim, _ = await kb_db.get_or_create_claim("fact", "Target claim with no persisted embedding.")
    other, _ = await kb_db.get_or_create_claim("fact", "Candidate claim with no persisted embedding either.")

    async def failing_embed_texts(*args, **kwargs):
        raise ConnectionError("simulated Ollama outage")

    monkeypatch.setattr(v, "embed_texts", failing_embed_texts)

    config = load_config()
    ranked = await v._rank_candidates_by_similarity(config, claim, [other])

    assert ranked == []  # degraded to "nothing ranked", not an exception


# -- run_verification_sweep concurrency guard --------------------------------


async def test_run_search_budget_is_atomic_and_bounded():
    budget = v._RunSearchBudget(2)

    assert await budget.reserve() is True
    assert await budget.reserve() is True
    assert await budget.reserve() is False
    assert budget.used == 2


async def test_run_search_budget_can_be_unbounded_for_explicit_actions():
    budget = v._RunSearchBudget(None)

    assert await budget.reserve() is True
    assert await budget.reserve() is True
    assert budget.used == 0


async def test_batch_verification_detects_the_model_once(monkeypatch):
    calls = {"detect": 0, "models": []}

    async def fake_detect(_url):
        calls["detect"] += 1
        return "shared-model"

    async def fake_verify(_db, _config, claim_id, **kwargs):
        calls["models"].append(kwargs["extraction_model"])
        return SimpleNamespace(status="supported")

    monkeypatch.setattr(v, "detect_model", fake_detect)
    monkeypatch.setattr(v, "verify_claim", fake_verify)

    outcomes = await v.verify_claims_concurrently(
        None, Config(), [{"id": "claim-1"}, {"id": "claim-2"}], concurrency=2,
    )

    assert len(outcomes) == 2
    assert calls == {"detect": 1, "models": ["shared-model", "shared-model"]}
# verify_claim makes real LLM calls against a single shared GPU (the machine
# this runs on has one, with a second coming later) -- a second sweep starting
# while one is already in progress would double up GPU load for no benefit,
# which is exactly what happened once in practice: the nightly cron fired
# while an orphaned manual-trigger run (from a killed dev server) was still
# marked "running" in the database.

async def test_run_verification_sweep_prioritizes_never_attempted_claims_over_retries(kb_db, monkeypatch):
    """Confirmed live: a service restart recomputes the eligible list from
    scratch, and force=True (needed to reopen the whole backlog) bypasses
    the retry cooldown -- so a high-importance claim that already failed to
    verify (still unverified) floats right back to the top of a fresh run
    and gets re-attempted again, while a never-tried, lower-importance claim
    is starved. One night of restarts left 99% of one run re-checking
    already-tried claims this way. Never-attempted claims must sort first
    regardless of importance/topics."""
    from deep_research.config import load_config

    retried_high_importance, _ = await kb_db.get_or_create_claim(
        "fact", "A high-importance claim already retried several times.", importance_score=0.95,
    )
    async with kb_db.pool.acquire() as conn:
        await conn.execute(
            "UPDATE claims SET verification_attempted_at = now() - interval '1 hour' WHERE id = $1",
            retried_high_importance["id"],
        )

    fresh_low_importance, _ = await kb_db.get_or_create_claim(
        "fact", "A low-importance claim never attempted before.", importance_score=0.1,
    )

    seen_order = []

    async def fake_verify_claims_concurrently(kb_db, config, claims, **kwargs):
        seen_order.extend(c["id"] for c in claims)
        return []

    monkeypatch.setattr(v, "verify_claims_concurrently", fake_verify_claims_concurrently)

    await v.run_verification_sweep(
        kb_db, load_config(), trigger="manual", force=True, only_status="unverified", threshold=0.0,
    )

    assert seen_order.index(fresh_low_importance["id"]) < seen_order.index(retried_high_importance["id"])


async def test_run_verification_sweep_refuses_concurrent_run(kb_db):
    import pytest

    await kb_db.create_verification_run("manual", claims_total=1)

    with pytest.raises(RuntimeError, match="already in progress"):
        await v.run_verification_sweep(kb_db, None, trigger="cron")


async def test_run_verification_sweep_treats_old_running_run_as_abandoned(kb_db):
    """A "running" row older than the cron job's own 8h timeout can only mean
    the process that owned it died without marking it complete -- it must not
    block new sweeps forever."""
    from deep_research.config import load_config

    stale = await kb_db.create_verification_run("cron", claims_total=1)
    async with kb_db.pool.acquire() as conn:
        await conn.execute(
            "UPDATE verification_runs SET started_at = started_at - INTERVAL '10 hours' WHERE id = $1",
            stale["id"],
        )

    config = load_config()
    summary = await v.run_verification_sweep(kb_db, config, trigger="cron", limit=0)

    stale_after = await kb_db.list_verification_runs(limit=10)
    stale_row = next(r for r in stale_after if r["id"] == stale["id"])
    assert stale_row["status"] == "failed"


async def test_run_verification_sweep_run_max_web_searches_overrides_config(kb_db, monkeypatch):
    """A one-off backlog sweep needs a much larger shared search budget than
    the routine nightly default, which would otherwise exhaust itself in the
    first few dozen of hundreds/thousands of claims."""
    from deep_research.config import load_config

    captured = {}

    class FakeBudget:
        def __init__(self, limit):
            captured["limit"] = limit
            self.limit = limit
            self.used = 0

    monkeypatch.setattr(v, "_RunSearchBudget", FakeBudget)
    config = load_config()

    await v.run_verification_sweep(kb_db, config, trigger="manual", limit=0, run_max_web_searches=2000)
    assert captured["limit"] == 2000


async def test_run_verification_sweep_defaults_run_max_web_searches_to_config(kb_db, monkeypatch):
    from deep_research.config import load_config

    captured = {}

    class FakeBudget:
        def __init__(self, limit):
            captured["limit"] = limit
            self.limit = limit
            self.used = 0

    monkeypatch.setattr(v, "_RunSearchBudget", FakeBudget)
    config = load_config()

    await v.run_verification_sweep(kb_db, config, trigger="manual", limit=0)
    assert captured["limit"] == config.kb.verification_run_max_web_searches


# -- keep-or-delete claims discovered during web-fallback verification ------
# The bug this guards against: extracting a scraped page's top chunks can
# promote several new claims, but only the one an LLM comparison pass
# actually classifies as supports/contradicts is worth keeping. Without
# discarding the rest, verifying claim A pulls in claims B/C/D from other
# pages, and if any of them also meet the importance threshold, the next
# sweep verifies *them* too -- unbounded compounding growth with no way to
# ever finish. Found in practice: a real KB had 1500+ such claims after one
# night, almost none of them ever established as relevant to anything, all
# floating with no topic and all eligible to keep the chain going.

async def _make_source_with_evidenced_claim(kb_db, claim_id, canonical_uri):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri=canonical_uri, canonical_key=canonical_uri,
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/x", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id=f"art-{canonical_uri}", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/x.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunk = await kb_db.add_chunk(artifact["id"], 0, "text", "chash")
    await kb_db.add_claim_evidence(
        claim_id=claim_id, artifact_chunk_id=chunk["id"], source_id=source["id"], source_version_id=version["id"],
    )
    return source


async def test_resolve_new_verification_claims_keeps_and_tags_the_proving_claim(kb_db):
    topic = await kb_db.create_topic("Data Centers")
    kept, _ = await kb_db.get_or_create_claim("fact", "Groundwater supply could soon come under pressure.")
    discarded, _ = await kb_db.get_or_create_claim("fact", "Tangential fact from the same scraped page.")
    source = await _make_source_with_evidenced_claim(kb_db, kept["id"], "http://kept-claim-source.example")

    await v._resolve_new_verification_claims(
        kb_db, [kept["id"], discarded["id"]], kept_ids={kept["id"]}, topics=[topic], source_id=source["id"],
    )

    kept_after = await kb_db.get_claim(kept["id"])
    assert kept_after["verification_override"] == "exclude"
    assert v.claim_check_status(kept_after, threshold=0.8) == "manual_exclude"
    linked_topics = await kb_db.get_topics_for_claim(kept["id"])
    assert [t["id"] for t in linked_topics] == [topic["id"]]

    # the source itself is tied to the topic too, not just the claim -- so
    # it shows up in context on the Sources page instead of floating
    # unexplained.
    source_topics = await kb_db.list_topic_sources(topic["id"])
    assert [s["id"] for s in source_topics] == [source["id"]]


async def test_resolve_new_verification_claims_deletes_the_non_proving_claims(kb_db):
    kept, _ = await kb_db.get_or_create_claim("fact", "The proving claim.")
    discarded, _ = await kb_db.get_or_create_claim("fact", "A tangential fact never established as relevant.")
    source = await _make_source_with_evidenced_claim(kb_db, kept["id"], "http://another-kept-claim-source.example")

    await v._resolve_new_verification_claims(
        kb_db, [kept["id"], discarded["id"]], kept_ids={kept["id"]}, topics=[], source_id=source["id"],
    )

    assert await kb_db.get_claim(kept["id"]) is not None
    assert await kb_db.get_claim(discarded["id"]) is None


# -- deleting sources that contributed nothing -------------------------------
# A page scraped/chunked/extracted purely to check one claim can end up with
# no surviving claim at all (empty page, nothing extractable, or the one
# claim it produced wasn't the prover/disprover) -- just as much dead weight
# as the discarded claims, and left unaddressed it accumulates the same way
# (187 sources found in a real KB, 166 with zero surviving claim_evidence).

async def test_source_has_claim_evidence_false_for_untouched_source(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="http://contributed-nothing.example", canonical_key="nothing",
    )
    assert await kb_db.source_has_claim_evidence(source["id"]) is False


async def test_delete_source_cascade_removes_source_and_its_artifacts(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="http://to-delete.example", canonical_key="to-delete",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/to-delete", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="art-to-delete", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/to-delete.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunk = await kb_db.add_chunk(artifact["id"], 0, "some text", "chash-1")

    assert await kb_db.source_has_claim_evidence(source["id"]) is False
    await kb_db.delete_source_cascade(source["id"])

    assert await kb_db.get_source(source["id"]) is None
    assert await kb_db.list_chunks(artifact["id"]) == []


# -- claims.verification_context: expands what verify_claim looks for --------

async def test_set_claim_verification_context_sets_and_clears(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "Industrial buildings use more electricity than residential.")

    updated = await kb_db.set_claim_verification_context(claim["id"], "compare against datacenter usage")
    assert updated["verification_context"] == "compare against datacenter usage"

    cleared = await kb_db.set_claim_verification_context(claim["id"], None)
    assert cleared["verification_context"] is None


async def test_set_claim_verification_context_strips_and_treats_blank_as_clear(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "Some claim.")

    updated = await kb_db.set_claim_verification_context(claim["id"], "  padded context  ")
    assert updated["verification_context"] == "padded context"

    blanked = await kb_db.set_claim_verification_context(claim["id"], "   ")
    assert blanked["verification_context"] is None


async def test_supporting_claims_are_durable_first_class_evidence(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "Main claim")
    support, _ = await kb_db.get_or_create_claim("fact", "Independent supporting claim")

    created = await kb_db.record_claim_supports(claim["id"], [support["id"], support["id"]])

    assert len(created) == 1
    assert await kb_db.get_claim_support_ids(claim["id"]) == [support["id"]]
