from deep_research.kb.canonical import is_social_media_domain, normalize_url


def test_recognizes_reddit_instagram_facebook_and_subdomains():
    assert is_social_media_domain("https://www.reddit.com/r/test/comments/abc") is True
    assert is_social_media_domain("https://old.reddit.com/r/test") is True
    assert is_social_media_domain("https://instagram.com/somepost") is True
    assert is_social_media_domain("https://www.facebook.com/somepage/posts/123") is True


def test_does_not_flag_unrelated_domains():
    assert is_social_media_domain("https://www.nytimes.com/2026/01/01/article.html") is False
    assert is_social_media_domain("https://en.wikipedia.org/wiki/Something") is False
    assert is_social_media_domain("https://notreddit.com/fake") is False


def test_normalize_url_strips_a_leading_www_prefix():
    assert normalize_url("https://www.example.com/page") == normalize_url("https://example.com/page")


def test_normalize_url_folds_arxiv_abstract_and_html_renderings():
    abs_url = "https://arxiv.org/abs/2401.12345"
    html_url = "https://arxiv.org/html/2401.12345v2"
    assert normalize_url(abs_url) == normalize_url(html_url)


def test_normalize_url_does_not_fold_arxiv_pdf_downloads():
    assert normalize_url("https://arxiv.org/abs/2401.12345") != normalize_url("https://arxiv.org/pdf/2401.12345")
