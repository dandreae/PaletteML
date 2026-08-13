"""Tests for the Art Institute of Chicago source (data/sources/artic.py).

No live network calls: `requests.Session` is replaced with a fake
that returns canned responses, so these run offline and
deterministically.
"""

from unittest.mock import Mock

import pytest
import requests

from paletteml.data.sources.artic import AicSource, ImageDownloadError

VALID_ITEM = {
    "id": 11,
    "title": "Self-Portrait",
    "artist_display": "Walter Shirlaw (American, 1838-1909)",
    "date_display": "1878",
    "date_start": 1878,
    "date_end": 1878,
    "is_public_domain": True,
    "artwork_type_title": "Painting",
    "image_id": "7b7a6f39-1cd8-ea2f-9811-18b0e23edac0",
}
NOT_PUBLIC_DOMAIN = {**VALID_ITEM, "id": 12, "is_public_domain": False}
NOT_A_PAINTING = {**VALID_ITEM, "id": 13, "artwork_type_title": "Sculpture"}
NO_IMAGE = {**VALID_ITEM, "id": 14, "image_id": None}


def _fake_search_response(items: list[dict]) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"data": items}
    return response


class TestIterCandidates:
    def test_filters_out_invalid_candidates(self):
        session = Mock()
        session.get.return_value = _fake_search_response(
            [VALID_ITEM, NOT_PUBLIC_DOMAIN, NOT_A_PAINTING, NO_IMAGE]
        )
        source = AicSource(session=session, request_delay=0)

        results = list(source.iter_candidates(limit=10))

        assert len(results) == 1
        assert results[0].artwork_id == "artic:11"
        assert results[0].title == "Self-Portrait"
        assert results[0].license == "CC0"
        assert results[0].source == "artic"
        assert results[0].image_url.endswith("/full/600,/0/default.jpg")

    def test_stops_at_limit_without_extra_requests(self):
        two_valid = [VALID_ITEM, {**VALID_ITEM, "id": 99}]
        session = Mock()
        session.get.return_value = _fake_search_response(two_valid)
        source = AicSource(session=session, request_delay=0)

        results = list(source.iter_candidates(limit=1))

        assert len(results) == 1
        assert session.get.call_count == 1  # didn't page further than needed

    def test_stops_when_source_exhausted(self):
        session = Mock()
        session.get.return_value = _fake_search_response([])  # empty page immediately
        source = AicSource(session=session, request_delay=0)

        results = list(source.iter_candidates(limit=50))

        assert results == []
        assert session.get.call_count == 1

    def test_dedupes_repeated_ids_across_pages(self):
        session = Mock()
        session.get.side_effect = [
            _fake_search_response([VALID_ITEM]),
            _fake_search_response([VALID_ITEM]),  # same id again
            _fake_search_response([]),
        ]
        source = AicSource(session=session, request_delay=0)

        results = list(source.iter_candidates(limit=10))

        assert len(results) == 1

    def test_paginates_when_first_page_insufficient(self):
        page1 = [NOT_PUBLIC_DOMAIN]  # yields nothing valid
        page2 = [VALID_ITEM]
        session = Mock()
        session.get.side_effect = [_fake_search_response(page1), _fake_search_response(page2)]
        source = AicSource(session=session, request_delay=0)

        results = list(source.iter_candidates(limit=1))

        assert len(results) == 1
        assert session.get.call_count == 2


class TestDownloadImage:
    def test_writes_response_content_to_dest_path(self, tmp_path):
        session = Mock()
        response = Mock()
        response.raise_for_status = Mock()
        response.content = b"fake-jpeg-bytes"
        session.get.return_value = response
        source = AicSource(session=session, request_delay=0)

        dest = tmp_path / "sub" / "artic_11.jpg"
        metadata = next(
            iter(
                AicSource(
                    session=Mock(get=Mock(return_value=_fake_search_response([VALID_ITEM]))),
                    request_delay=0,
                ).iter_candidates(1)
            )
        )
        source.download_image(metadata, dest)

        assert dest.read_bytes() == b"fake-jpeg-bytes"
        called_headers = session.get.call_args.kwargs["headers"]
        assert "AIC-User-Agent" in called_headers

    def test_retries_then_raises_image_download_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("paletteml.data.sources.artic.time.sleep", lambda _: None)
        session = Mock()
        session.get.side_effect = requests.ConnectionError("boom")
        source = AicSource(session=session, request_delay=0)

        metadata = next(
            iter(
                AicSource(
                    session=Mock(get=Mock(return_value=_fake_search_response([VALID_ITEM]))),
                    request_delay=0,
                ).iter_candidates(1)
            )
        )

        with pytest.raises(ImageDownloadError):
            source.download_image(metadata, tmp_path / "artic_11.jpg")
        assert session.get.call_count == 3  # MAX_DOWNLOAD_RETRIES
