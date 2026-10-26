"""Synthetic state-ownership regressions. No Bluetooth hardware is used."""

import asyncio
import base64
import hashlib
import unittest

from bleak.backends.scanner import AdvertisementData

from eufylife_ble_client import util
from eufylife_ble_client.auth_handler import AuthHandler, IV
from eufylife_ble_client.client import EufyLifeBLEDevice
from eufylife_ble_client.models import EufyLifeBLEState


ADDRESS = "00:00:00:00:00:01"


def advertisement(weight):
    """Build a synthetic final-weight advertisement for a T9130."""
    data = bytearray(18)
    data[10] = 0x05
    data[12:14] = weight.to_bytes(2, "little")
    return AdvertisementData(
        local_name="eufy T9130",
        manufacturer_data={0: bytes(data)},
        service_data={},
        service_uuids=[],
        tx_power=None,
        rssi=-60,
        platform_data=(),
    )


def c1_segments(device_uuid):
    """Encrypt and frame a synthetic reply using the existing protocol helpers."""
    key = hashlib.md5(ADDRESS.replace(":", "").encode("utf-8")).digest()
    encrypted = util.encrypt_aes128cbc(device_uuid.encode("utf-8"), key, IV)
    encoded = base64.b64encode(encrypted).hex()
    return [bytes.fromhex(segment) for segment in util.get_sub_contract_bytes(encoded, "C1")]


class DeviceCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_updates_only_notify_the_owning_device(self):
        first = EufyLifeBLEDevice("eufy T9130")
        first_seen = []
        unregister_first = first.register_callback(first_seen.append)
        self.addCleanup(unregister_first)

        # Creating another device must neither clear nor share the first registry.
        second = EufyLifeBLEDevice("eufy T9130")
        second_seen = []
        self.addCleanup(second.register_callback(second_seen.append))

        first.update_state_from_advertisement_data(advertisement(1000))
        self.assertEqual(first_seen, [EufyLifeBLEState(10.0, 10.0, None, False)])
        self.assertEqual(second_seen, [])
        self.assertEqual(second.state, EufyLifeBLEState())

        second.update_state_from_advertisement_data(advertisement(2000))
        self.assertEqual(second_seen, [EufyLifeBLEState(20.0, 20.0, None, False)])
        self.assertEqual(len(first_seen), 1)
        self.assertIs(first_seen[0], first.state)
        self.assertIs(second_seen[0], second.state)

        first.update_state_from_advertisement_data(advertisement(1000))
        self.assertEqual(len(first_seen), 1)
        self.assertEqual(len(second_seen), 1)

    async def test_unregister_does_not_expose_another_devices_callback(self):
        first = EufyLifeBLEDevice("eufy T9130")
        second = EufyLifeBLEDevice("eufy T9130")
        seen = []
        # The same callable can be registered independently on both devices.
        callback = seen.append
        unregister_first = first.register_callback(callback)
        self.addCleanup(second.register_callback(callback))
        unregister_first()

        first.update_state_from_advertisement_data(advertisement(1000))
        self.assertEqual(seen, [])
        second.update_state_from_advertisement_data(advertisement(2000))
        self.assertEqual(seen, [second.state])
        self.assertEqual(second.state.weight_kg, 20.0)


class AuthHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_handlers_use_their_own_c1_reply(self):
        for device_uuid in ("fixture-one", "fixture-two"):
            handler = AuthHandler(ADDRESS, asyncio.get_running_loop())
            handler.generate_key()
            segments = c1_segments(device_uuid)
            self.assertGreater(len(segments), 1)
            self.assertFalse(handler.c1_future.done())
            for segment in segments[:-1]:
                handler.handle_c1(segment)
                self.assertFalse(handler.c1_future.done())
            handler.handle_c1(segments[-1])
            self.assertTrue(handler.c1_future.done())
            self.assertEqual(handler.c1_future.result(), device_uuid)

    async def test_interleaved_handlers_keep_partial_c1_replies_separate(self):
        first = AuthHandler(ADDRESS, asyncio.get_running_loop())
        first.generate_key()
        first_segments = c1_segments("fixture-one")
        first.handle_c1(first_segments[0])
        self.assertFalse(first.c1_future.done())

        # A new session must not reset or append to the first session's buffer.
        second = AuthHandler(ADDRESS, asyncio.get_running_loop())
        second.generate_key()
        second_segments = c1_segments("fixture-two")
        second.handle_c1(second_segments[0])
        self.assertFalse(first.c1_future.done())
        self.assertFalse(second.c1_future.done())

        for segment in first_segments[1:]:
            first.handle_c1(segment)
        self.assertTrue(first.c1_future.done())
        self.assertEqual(first.c1_future.result(), "fixture-one")
        self.assertFalse(second.c1_future.done())

        for segment in second_segments[1:]:
            second.handle_c1(segment)
        self.assertTrue(second.c1_future.done())
        self.assertEqual(second.c1_future.result(), "fixture-two")
        self.assertEqual(first.c1_future.result(), "fixture-one")
