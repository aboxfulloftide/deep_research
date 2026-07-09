import httpx
import pytest

from deep_research.retry import with_retries


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://example.invalid")
    resp = httpx.Response(status_code, request=req)
    return httpx.HTTPStatusError("error", request=req, response=resp)


async def test_succeeds_immediately_without_retrying():
    calls = []

    async def ok():
        calls.append(1)
        return "success"

    result = await with_retries(ok)
    assert result == "success"
    assert len(calls) == 1


async def test_recovers_after_transient_connect_errors():
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://x"))
        return "recovered"

    result = await with_retries(flaky, attempts=5, base_delay=0.001)
    assert result == "recovered"
    assert attempts["n"] == 3


@pytest.mark.parametrize("status_code", [502, 503, 504])
async def test_retries_transient_5xx_status_codes(status_code):
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise _http_error(status_code)
        return "ok"

    result = await with_retries(flaky, attempts=3, base_delay=0.001)
    assert result == "ok"
    assert attempts["n"] == 2


async def test_non_transient_4xx_raises_immediately_without_retry():
    attempts = {"n": 0}

    async def bad_request():
        attempts["n"] += 1
        raise _http_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(bad_request, attempts=5, base_delay=0.001)
    assert attempts["n"] == 1


async def test_persistent_transient_failure_raises_after_exhausting_attempts():
    attempts = {"n": 0}

    async def always_down():
        attempts["n"] += 1
        raise httpx.ConnectTimeout("timeout", request=httpx.Request("GET", "http://x"))

    with pytest.raises(httpx.ConnectTimeout):
        await with_retries(always_down, attempts=3, base_delay=0.001)
    assert attempts["n"] == 3


async def test_non_httpx_exception_is_never_treated_as_transient():
    attempts = {"n": 0}

    async def raises_value_error():
        attempts["n"] += 1
        raise ValueError("not a network error")

    with pytest.raises(ValueError):
        await with_retries(raises_value_error, attempts=5, base_delay=0.001)
    assert attempts["n"] == 1
