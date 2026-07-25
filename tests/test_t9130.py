"""Tests for Eufy Smart Scale C20 (T9130) advertisements.

The status sequence and frame lengths are backed by a private HA capture. The
measurement values remain synthetic and contain no user health data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eufylife_ble_client.client import EufyLifeBLEDevice, _parse_t9130_advertisement
from eufylife_ble_client.models import EufyLifeBLEState

LIVE_ID = 48205
BODY_ID = 48207
FINAL_STATUSES = {0x05, 0x15, 0x25, 0x65, 0xA5}


def _packet(
    *,
    status: int,
    weight_kg: float = 80.0,
    heart_rate: int = 0,
    impedance_ohm: float = 0.0,
    length: int = 19,
) -> bytearray:
    """Build a synthetic T9130 manufacturer-data payload."""
    data = bytearray(length)
    if length > 10:
        data[10] = status

    if length > 13:
        weight_raw = round(weight_kg * 100)
        data[12] = weight_raw % 256
        data[13] = weight_raw // 256

    if length > 15:
        data[15] = heart_rate

    if length > 18:
        impedance_raw = round(impedance_ohm * 10)
        data[17] = impedance_raw % 256
        data[18] = impedance_raw // 256

    return data


def _device(model: str = "eufy T9130") -> EufyLifeBLEDevice:
    device = object.__new__(EufyLifeBLEDevice)
    device._model_id = model
    device._state = EufyLifeBLEState()
    device._callbacks = []
    return device


def test_in_progress_weight() -> None:
    state = _parse_t9130_advertisement(_packet(status=0x01))

    assert state is not None
    assert state.weight_kg == pytest.approx(80.0)
    assert state.final_weight_kg is None
    assert state.heart_rate is None
    assert state.impedance_ohm is None


@pytest.mark.parametrize("status", [0x05, 0x15, 0x25])
def test_observed_final_weight_statuses(status: int) -> None:
    state = _parse_t9130_advertisement(_packet(status=status))

    assert state is not None
    assert state.final_weight_kg == pytest.approx(80.0)
    assert state.heart_rate is None
    assert state.impedance_ohm is None


def test_observed_impedance_status() -> None:
    state = _parse_t9130_advertisement(_packet(status=0x65, impedance_ohm=512.3))

    assert state is not None
    assert state.final_weight_kg == pytest.approx(80.0)
    assert state.impedance_ohm == pytest.approx(512.3)
    assert state.heart_rate is None


def test_observed_heart_rate_status_keeps_persistent_impedance_bytes() -> None:
    state = _parse_t9130_advertisement(
        _packet(status=0xA5, heart_rate=72, impedance_ohm=512.3, length=24)
    )

    assert state is not None
    assert state.final_weight_kg == pytest.approx(80.0)
    assert state.heart_rate == 72
    assert state.impedance_ohm == pytest.approx(512.3)


@pytest.mark.parametrize("status", [0x00, 0x03, 0x04, 0x0D, 0x45, 0x85, 0xC5])
def test_unobserved_statuses_fail_closed(status: int) -> None:
    assert _parse_t9130_advertisement(_packet(status=status)) is None


@pytest.mark.parametrize("length", [0, 17])
def test_short_packets_are_ignored(length: int) -> None:
    assert _parse_t9130_advertisement(bytearray(length)) is None


def test_short_body_packet_is_ignored_without_index_error() -> None:
    assert _parse_t9130_advertisement(_packet(status=0x65, length=18)) is None
    assert _parse_t9130_advertisement(_packet(status=0xA5, length=18)) is None


def test_zero_weight_is_ignored() -> None:
    assert _parse_t9130_advertisement(_packet(status=0x01, weight_kg=0.0)) is None


def test_device_prefers_body_frame_and_fires_once() -> None:
    device = _device()
    observed: list[EufyLifeBLEState] = []
    device._callbacks = [observed.append]
    advertisement = SimpleNamespace(
        manufacturer_data={
            LIVE_ID: _packet(status=0x01, weight_kg=80.5, length=18),
            BODY_ID: _packet(
                status=0x65,
                weight_kg=80.0,
                impedance_ohm=512.3,
                length=24,
            ),
        }
    )

    device.update_state_from_advertisement_data(advertisement)

    assert len(observed) == 1
    assert device.state.weight_kg == pytest.approx(80.0)
    assert device.state.final_weight_kg == pytest.approx(80.0)
    assert device.state.impedance_ohm == pytest.approx(512.3)


def test_live_frame_selection_is_deterministic_when_order_is_reversed() -> None:
    device = _device()
    advertisement = SimpleNamespace(
        manufacturer_data={
            BODY_ID: _packet(status=0x01, weight_kg=80.0, length=18),
            LIVE_ID: _packet(status=0x01, weight_kg=80.5, length=18),
        }
    )

    device.update_state_from_advertisement_data(advertisement)

    assert device.state.weight_kg == pytest.approx(80.5)


def test_foreign_manufacturer_data_is_ignored() -> None:
    device = _device()
    advertisement = SimpleNamespace(
        manufacturer_data={
            0x0001: _packet(
                status=0xA5,
                weight_kg=90.0,
                heart_rate=99,
                impedance_ohm=700.0,
                length=24,
            ),
            LIVE_ID: _packet(status=0x01, weight_kg=80.5, length=18),
        }
    )

    device.update_state_from_advertisement_data(advertisement)

    assert device.state.weight_kg == pytest.approx(80.5)
    assert device.state.final_weight_kg is None


def test_final_packet_is_self_contained_and_does_not_cross_session_latch() -> None:
    device = _device()
    device.update_state_from_advertisement_data(
        SimpleNamespace(
            manufacturer_data={
                BODY_ID: _packet(
                    status=0xA5,
                    weight_kg=80.0,
                    heart_rate=72,
                    impedance_ohm=512.3,
                    length=24,
                )
            }
        )
    )
    assert device.state.heart_rate == 72
    assert device.state.impedance_ohm == pytest.approx(512.3)

    device.update_state_from_advertisement_data(
        SimpleNamespace(
            manufacturer_data={BODY_ID: _packet(status=0x05, weight_kg=80.0, length=18)}
        )
    )

    assert device.state.final_weight_kg == pytest.approx(80.0)
    assert device.state.heart_rate is None
    assert device.state.impedance_ohm is None


def test_capture_backed_synthetic_sequence_has_no_stale_final() -> None:
    fixture_path = Path(__file__).parent / "fixtures/t9130_sequence_synthetic.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    device = _device()
    observed: list[EufyLifeBLEState] = []
    device._callbacks = [observed.append]
    callback_records: list[tuple[int, EufyLifeBLEState]] = []

    expected_first_final = next(
        record["sequence"]
        for record in fixture["records"]
        if bytes.fromhex(record["manufacturer_data"][str(BODY_ID)])[10]
        in FINAL_STATUSES
    )

    for record in fixture["records"]:
        before = len(observed)
        manufacturer_data = {
            int(key): bytearray.fromhex(value)
            for key, value in record["manufacturer_data"].items()
        }
        device.update_state_from_advertisement_data(
            SimpleNamespace(manufacturer_data=manufacturer_data)
        )
        callback_records.extend(
            (record["sequence"], state) for state in observed[before:]
        )

    first_observed_final = next(
        sequence
        for sequence, state in callback_records
        if state.final_weight_kg is not None
    )
    assert first_observed_final == expected_first_final
    assert all(
        state.final_weight_kg is None
        for sequence, state in callback_records
        if sequence < expected_first_final
    )
    assert device.state.weight_kg == pytest.approx(80.0)
    assert device.state.final_weight_kg == pytest.approx(80.0)
    assert device.state.impedance_ohm == pytest.approx(512.3)
    assert device.state.heart_rate == 72


def test_t9150_multiple_entry_selection_is_unchanged() -> None:
    device = _device("eufy T9150")
    valid = _packet(status=0x05, weight_kg=81.0, length=18)

    device.update_state_from_advertisement_data(
        SimpleNamespace(manufacturer_data={0x0001: bytearray(18), 0x0002: valid})
    )

    assert device.state.weight_kg == pytest.approx(81.0)
    assert device.state.final_weight_kg == pytest.approx(81.0)


def test_t9130_capabilities_are_exposed() -> None:
    device = _device()

    assert device.supports_heart_rate
    assert device.supports_impedance
