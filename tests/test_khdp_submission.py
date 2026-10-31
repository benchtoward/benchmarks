import httpx

from benchtoward.khdp import KhdpClient, submission_label


def _client(handler):
    return KhdpClient("khdp_pat_test", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_submission_status_and_doi():
    def handler(request):
        assert request.headers["authorization"] == "Bearer khdp_pat_test"
        if request.url.path == "/submissions/my-bench/1.0.0":
            return httpx.Response(200, json={"code": "my-bench", "version": "1.0.0", "status": "published",
                                             "title": "My bench", "doiLink": "https://doi.org/10.23421/2026.09999"})
        return httpx.Response(404, json={"message": "not found"})

    s = _client(handler).submission("my-bench", "1.0.0")
    assert (s.code, s.version, s.status, s.doi) == ("my-bench", "1.0.0", "published", "10.23421/2026.09999")
    assert _client(handler).submission("other", "1.0.0") is None


def test_submission_labels():
    assert submission_label("draft") == "draft"
    assert submission_label("admin_review") == "under review"
    assert submission_label("published") == "published"
    assert submission_label("rejected") == "rejected"
