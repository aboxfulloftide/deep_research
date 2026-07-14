from deep_research.models import SearchResult
from deep_research.tools.search import _rank_results


def _result(title: str, snippet: str = "") -> SearchResult:
    return SearchResult(title=title, url=f"https://example.test/{title}", snippet=snippet)


def test_rank_results_prefers_results_about_the_question_over_stale_results():
    query = "Did Donald Trump say racists were very fine people?"
    results = [
        _result("Dissociative Identity Disorder", "A medical condition with multiple identities."),
        _result("Trump's very fine people comments", "Donald Trump discussed Charlottesville."),
    ]

    ranked = _rank_results(results, query)

    assert ranked[0].title == "Trump's very fine people comments"


def test_rank_results_keeps_single_term_queries_searchable():
    result = _result("Qwen language model", "Qwen is a family of large language models.")

    assert _rank_results([result], "Qwen") == [result]
