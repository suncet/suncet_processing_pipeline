"""Collector recovery and privacy tests; no network or production DB writes."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from suncet_processing_pipeline.satnogs import collect as c
from suncet_processing_pipeline.satnogs.ax25 import SUNCET_AX25_HEADER

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)
PACKET = bytes.fromhex((Path(c.__file__).parent / "test_data/suncet_apid1_synthetic_252.hex").read_text())
CHECK_STORAGE = c.check_storage


def test_example_config_defaults_token_to_service_home_and_allows_override(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    example = Path(__file__).resolve().parents[2] / "operations/satnogs.example.ini"
    config = c.Config.read(example)
    assert config.token_file == tmp_path / ".config/suncet/satnogs.token"
    custom = tmp_path / "custom.ini"
    token = tmp_path / "custom.token"
    custom.write_text(example.read_text().replace(
        "[satnogs]", f"[satnogs]\ntoken_file = {token}"))
    assert c.Config.read(custom).token_file == token


def record(**overrides):
    return {"sat_id": c.SAT_ID, "timestamp": "2026-10-20T12:00:00Z",
            "frame": (SUNCET_AX25_HEADER + PACKET).hex(), "observer": "synthetic",
            "station_id": 1, "observation_id": 2, "decoded": "influxdb", **overrides}


def page(rows=(), next_url=None):
    return json.dumps({"results": list(rows), "next": next_url, "previous": None}).encode()


class FakeClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "check_storage", lambda config: None)
    return c.Config(tmp_path / "private", tmp_path / "token", NOW - timedelta(days=30), tmp_path)


def test_pages_decode_and_repeated_download_preserves_station_receptions(config):
    following = c.ENDPOINT + "?cursor=next"
    rows = [record(), record(station_id=3)]
    result = c.collect(config, FakeClient(page(rows[:1], following), page(rows[1:])), now=NOW)
    assert result["new_receipts"] == 2
    assert result["pages"] == 2
    again = c.collect(config, FakeClient(page(rows)), now=NOW + timedelta(days=1))
    assert again["new_receipts"] == 0
    assert again["duplicates"] == 2
    assert c.status(config)["accepted"] == 2
    with sqlite3.connect(config.directory / "telemetry.sqlite3") as connection:
        values = json.loads(connection.execute("SELECT decoded_json FROM receipts LIMIT 1").fetchone()[0])
        assert len(values) == 111
        assert not any("command" in key for key in values)
        assert connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 3


def test_partial_failure_keeps_receipts_but_does_not_advance_checkpoint(config):
    c.collect(config, FakeClient(page()), now=NOW)
    client = FakeClient(page([record()], c.ENDPOINT + "?cursor=2"), c.CollectionError("offline"))
    with pytest.raises(c.CollectionError, match="offline"):
        c.collect(config, client, now=NOW + timedelta(days=1))
    assert c.status(config)["last_success"] == c.iso(NOW)
    assert c.status(config)["receipts"] == 1
    retry = c.collect(config, FakeClient(page([record()])), now=NOW + timedelta(days=2))
    assert retry["duplicates"] == 1


def test_empty_success_overlap_and_monthly_reconciliation(config):
    first = FakeClient(page())
    c.collect(config, first, now=NOW)
    assert parse_qs(urlsplit(first.urls[0]).query)["start"] == [c.iso(config.initial_start)]
    second = FakeClient(page([record(timestamp="2026-10-19T00:00:00Z")]))
    c.collect(config, second, now=NOW + timedelta(days=1))
    assert parse_qs(urlsplit(second.urls[0]).query)["start"] == [c.iso(NOW - timedelta(days=7))]
    third = FakeClient(page())
    c.collect(config, third, now=NOW + timedelta(days=31))
    assert parse_qs(urlsplit(third.urls[0]).query)["start"] == [c.iso(config.initial_start)]


def test_one_off_historical_pull_does_not_skip_daily_window(config):
    c.collect(config, FakeClient(page()), now=NOW)
    c.collect(config, FakeClient(page()), now=NOW + timedelta(days=1), since=NOW)
    assert c.status(config)["last_success"] == c.iso(NOW)


def test_corrupt_or_other_packet_is_retained_without_decoded_values(config):
    c.collect(config, FakeClient(page([record(frame="00"), record(frame=PACKET.hex())])), now=NOW)
    result = c.status(config)
    assert result["rejected"] == 2
    assert result["accepted"] == 0


@pytest.mark.parametrize("bad_page", [b"not json", page([record(sat_id="other")]),
                                      b'{"results": [], "next": ""}', page([record(timestamp="bad")])])
def test_bad_response_does_not_mark_success(config, bad_page):
    with pytest.raises(c.CollectionError):
        c.collect(config, FakeClient(bad_page), now=NOW)
    assert "last_success" not in c.status(config)


def test_pagination_loop_and_budget_stop_without_checkpoint(config):
    following = c.ENDPOINT + "?cursor=repeat"
    with pytest.raises(c.CollectionError, match="Pagination"):
        c.collect(config, FakeClient(page([], following), page([], following)), now=NOW)
    with pytest.raises(c.CollectionError, match="page limit"):
        c.collect(replace(config, max_pages=1), FakeClient(page([], following)), now=NOW)
    assert "last_success" not in c.status(config)


@pytest.mark.parametrize("url", ["https://evil.example/api/telemetry/", "http://db.satnogs.org/api/telemetry/",
                                  "https://db.satnogs.org@evil.example/api/telemetry/",
                                  "https://db.satnogs.org/other/"])
def test_pagination_cannot_exfiltrate_credentials(url):
    with pytest.raises(c.CollectionError, match="Unsafe"):
        c.Client("secret").get(url)


@pytest.mark.parametrize("code", [401, 403, 429])
def test_auth_and_throttle_fail_without_retry_or_secret_in_error(monkeypatch, code):
    client = c.Client("verysecret")
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        assert request.get_method() == "GET"
        raise HTTPError(c.ENDPOINT, code, "verysecret", {}, None)

    monkeypatch.setattr(client.opener, "open", fail)
    with pytest.raises(c.CollectionError) as error:
        client.get(c.ENDPOINT)
    assert "verysecret" not in str(error.value)
    assert len(calls) == 1


def test_redirects_are_not_followed():
    with pytest.raises(c.CollectionError, match="redirect"):
        c.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example")


def test_token_permissions_and_nonempty_value(tmp_path):
    path = tmp_path / "token"
    path.write_text("abc123\n")
    path.chmod(0o644)
    with pytest.raises(c.CollectionError, match="0600"):
        c.read_token(path)
    path.chmod(0o600)
    assert c.read_token(path) == "abc123"
    path.write_text("")
    with pytest.raises(c.CollectionError, match="token"):
        c.read_token(path)


def test_concurrent_run_is_refused_and_lock_releases(config):
    with c.locked_database(config.directory):
        with pytest.raises(c.CollectionError, match="already running"):
            c.collect(config, FakeClient(page()), now=NOW)
    c.collect(config, FakeClient(page()), now=NOW)


def test_mount_guard_stops_before_creating_archive(tmp_path):
    config = c.Config(tmp_path / "archive", tmp_path / "token", NOW, tmp_path)
    with pytest.raises(c.CollectionError, match="not mounted"):
        c.check_storage(config)
    assert not config.directory.exists()


def test_disk_guard_stops_before_any_request(config, monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda self: True)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 1})())
    with pytest.raises(c.CollectionError, match="free-space"):
        CHECK_STORAGE(config)


def test_real_client_paces_pages_and_preserves_request_filters(config, monkeypatch):
    client = c.Client("secret")
    calls, sleeps = [], []
    responses = iter([page([record()], c.ENDPOINT + "?cursor=next"), page()])

    def respond(request, **kwargs):
        calls.append(request)
        assert request.get_header("Authorization") == "Token secret"
        assert request.get_method() == "GET"
        return BytesIO(next(responses))

    monkeypatch.setattr(client.opener, "open", respond)
    monkeypatch.setattr(c.time, "monotonic", lambda: 100)
    monkeypatch.setattr(c.time, "sleep", sleeps.append)
    c.collect(config, client, now=NOW)
    query = parse_qs(urlsplit(calls[0].full_url).query)
    assert query["sat_id"] == [c.SAT_ID]
    assert query["end"] == [c.iso(NOW)]
    assert sleeps == [11]


def test_transient_server_errors_have_bounded_retries(monkeypatch):
    client = c.Client("secret")
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        raise HTTPError(c.ENDPOINT, 503, "unavailable", {}, None)

    monkeypatch.setattr(client.opener, "open", fail)
    monkeypatch.setattr(c.time, "sleep", lambda seconds: None)
    with pytest.raises(c.CollectionError, match="503"):
        client.get(c.ENDPOINT)
    assert len(calls) == 3


def test_cli_status_needs_no_token_and_archive_is_private(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c.Config, "read", lambda path: config)
    old_umask = c.os.umask(0o077)
    try:
        assert c.main(["status"]) == 0
        assert json.loads(capsys.readouterr().out)["receipts"] == 0
        c.collect(config, FakeClient(page([record()])), now=NOW)
        assert (config.directory / "telemetry.sqlite3").stat().st_mode & 0o077 == 0
        assert c.main(["status"]) == 0
        assert json.loads(capsys.readouterr().out)["accepted"] == 1
    finally:
        c.os.umask(old_umask)
