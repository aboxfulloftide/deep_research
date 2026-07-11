from deep_research.kb.canonical import is_social_media_domain


def test_recognizes_reddit_instagram_facebook_and_subdomains():
    assert is_social_media_domain("https://www.reddit.com/r/test/comments/abc") is True
    assert is_social_media_domain("https://old.reddit.com/r/test") is True
    assert is_social_media_domain("https://instagram.com/somepost") is True
    assert is_social_media_domain("https://www.facebook.com/somepage/posts/123") is True


def test_does_not_flag_unrelated_domains():
    assert is_social_media_domain("https://www.nytimes.com/2026/01/01/article.html") is False
    assert is_social_media_domain("https://en.wikipedia.org/wiki/Something") is False
    assert is_social_media_domain("https://notreddit.com/fake") is False
