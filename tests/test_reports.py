import httpx
import pytest

from deep_research.kb import reports as rpt


def _context_exceeded_error() -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://example.invalid")
    resp = httpx.Response(
        400,
        json={"error": {"code": 400, "message": "exceeds context", "type": "exceed_context_size_error"}},
        request=req,
    )
    return httpx.HTTPStatusError("exceeds context", request=req, response=resp)


def _generic_400_error() -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://example.invalid")
    resp = httpx.Response(400, json={"error": {"code": 400, "message": "bad request", "type": "invalid_request"}}, request=req)
    return httpx.HTTPStatusError("bad request", request=req, response=resp)


def test_is_context_exceeded_error_detects_the_specific_type():
    assert rpt._is_context_exceeded_error(_context_exceeded_error()) is True


def test_is_context_exceeded_error_rejects_other_400s():
    assert rpt._is_context_exceeded_error(_generic_400_error()) is False


def test_is_context_exceeded_error_rejects_non_http_errors():
    assert rpt._is_context_exceeded_error(ValueError("not an http error")) is False


def test_batch_blocks_returns_lists_not_joined_strings():
    batches = rpt._batch_blocks(["a", "b", "c"], budget_chars=1000)
    assert batches == [["a", "b", "c"]]
    assert all(isinstance(b, list) for b in batches)


class _FakeLLM:
    """Simulates a batch call that fails with exceed_context_size_error until
    the block list has been bisected down to a small enough piece, then
    succeeds -- proving _summarize_batch actually splits and recovers instead
    of failing the whole report."""

    def __init__(self, fails_above_block_count: int):
        self.fails_above_block_count = fails_above_block_count
        self.calls: list[int] = []

    async def chat(self, messages):
        content = messages[1]["content"]
        body = content.split("\n\n", 1)[1] if "\n\n" in content else content  # strip "Topic: x\n\n" prefix
        block_count = len(body.split("\n\n")) if body else 0
        self.calls.append(block_count)
        if block_count > self.fails_above_block_count:
            raise _context_exceeded_error()
        return {"choices": [{"message": {"content": f"summary of {block_count} block(s)"}}]}


async def test_summarize_batch_bisects_on_context_exceeded_and_recovers():
    llm = _FakeLLM(fails_above_block_count=1)
    blocks = ["block one", "block two", "block three", "block four"]

    result = await rpt._summarize_batch(llm, "Test Topic", blocks)

    assert "summary of 1 block(s)" in result
    # First attempt uses the whole batch (4 blocks) and fails; bisection then
    # keeps splitting until every sub-batch is down to 1 block, which succeeds.
    assert llm.calls[0] == 4
    assert len(llm.calls) > 1  # more than one call means bisection actually happened
    assert min(llm.calls) == 1  # eventually got down to single blocks, which succeeded


async def test_summarize_batch_does_not_bisect_on_unrelated_400():
    class RaisesGeneric400:
        async def chat(self, messages):
            raise _generic_400_error()

    with pytest.raises(httpx.HTTPStatusError):
        await rpt._summarize_batch(RaisesGeneric400(), "Test Topic", ["a", "b"])


async def test_summarize_batch_gives_up_on_single_unsplittable_block():
    class AlwaysFails:
        async def chat(self, messages):
            raise _context_exceeded_error()

    with pytest.raises(httpx.HTTPStatusError):
        await rpt._summarize_batch(AlwaysFails(), "Test Topic", ["one impossibly large block"])
