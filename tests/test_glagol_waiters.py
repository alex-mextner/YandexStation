"""Exercise real Glagol send/receive cancellation with an in-memory WebSocket."""

import asyncio
import importlib
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

# Import the real core without HA setup or adding calendar.py to sys.path.
core_package = ModuleType("glagol_test_core")
core_package.__path__ = [
    str(
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "yandex_station"
        / "core"
    )
]
sys.modules[core_package.__name__] = core_package
YandexGlagol = importlib.import_module("glagol_test_core.yandex_glagol").YandexGlagol


class FakeWebSocket:
    def __init__(self):
        self.closed = False
        self.incoming = asyncio.Queue()
        self.sent = asyncio.Queue()
        self.send_gate = None
        self.send_error = None

    async def send_json(self, data):
        if self.send_error is not None:
            raise self.send_error
        await self.sent.put(data)
        if self.send_gate is not None:
            await self.send_gate.wait()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return SimpleNamespace(data=json.dumps(await self.incoming.get()))

    async def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, ws):
        self.ws = ws
        self.connections = 0

    async def ws_connect(self, *_args, **_kwargs):
        self.connections += 1
        return self.ws


class GlagolWaiterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = FakeWebSocket()
        self.session = FakeSession(self.ws)
        self.glagol = YandexGlagol(self.session, {"name": "Regression speaker"})
        self.glagol.device_token = "fixture-token"
        self.glagol.url = "wss://speaker.invalid:1961"
        self.updates = []
        self.received = asyncio.Queue()
        self.requests = []

        def on_update(data):
            self.updates.append(data)
            self.received.put_nowait(data)

        self.glagol.update_handler = on_update
        self.connection = asyncio.create_task(self.glagol._connect(0))
        ping = await asyncio.wait_for(self.ws.sent.get(), 1)
        self.assertEqual(ping["payload"]["command"], "softwareVersion")

    async def asyncTearDown(self):
        self.glagol.url = None
        for task in [*self.requests, self.connection]:
            task.cancel()
        await asyncio.gather(*self.requests, self.connection, return_exceptions=True)
        self.glagol.waiters.clear()

    async def start_request(self):
        task = asyncio.create_task(self.glagol.send({"command": "fixture-command"}))
        self.requests.append(task)
        request = await asyncio.wait_for(self.ws.sent.get(), 1)
        return task, request["id"]

    async def reply(self, request_id, vins_response=None):
        data = {
            "requestId": request_id,
            "status": "SUCCESS",
            "state": {"aliceState": "IDLE"},
        }
        if vins_response is not None:
            data["vinsResponse"] = vins_response
        await self.ws.incoming.put(data)
        observed = await asyncio.wait_for(self.received.get(), 1)
        self.assertEqual(observed, data)
        self.assertFalse(self.connection.done())
        self.assertEqual(self.session.connections, 1)
        self.assertNotIn(None, self.updates)

    async def assert_next_request_works(self):
        task, request_id = await self.start_request()
        await self.reply(request_id)
        self.assertEqual(await asyncio.wait_for(task, 1), {"status": "SUCCESS"})
        self.assertNotIn(request_id, self.glagol.waiters)

    async def test_cancel_after_dispatch_keeps_connection_for_late_reply(self):
        task, request_id = await self.start_request()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn(request_id, self.glagol.waiters)
        await self.reply(request_id)
        await self.assert_next_request_works()

    async def test_reply_during_send_is_not_lost(self):
        self.ws.send_gate = asyncio.Event()
        task, request_id = await self.start_request()
        await self.reply(request_id)
        self.ws.send_gate.set()
        self.assertEqual(await asyncio.wait_for(task, 1), {"status": "SUCCESS"})
        self.assertNotIn(request_id, self.glagol.waiters)

    async def test_cancel_during_send_cleans_waiter_and_accepts_late_state(self):
        self.ws.send_gate = asyncio.Event()
        task, request_id = await self.start_request()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn(request_id, self.glagol.waiters)
        await self.reply(request_id)
        self.ws.send_gate = None
        await self.assert_next_request_works()

    async def test_timeout_cleanup_ignores_late_reply(self):
        real_wait_for = asyncio.wait_for
        timeout_seen = []

        async def expire_waiter(awaitable, timeout):
            timeout_seen.append(timeout)
            return await real_wait_for(awaitable, 0)

        with patch("glagol_test_core.yandex_glagol.asyncio.wait_for", expire_waiter):
            result = await self.glagol.send({"command": "fixture-timeout"})
        request = self.ws.sent.get_nowait()
        self.assertEqual(timeout_seen, [5])
        self.assertIn("error", result)
        self.assertNotIn(request["id"], self.glagol.waiters)
        await self.reply(request["id"])
        await self.assert_next_request_works()

    async def test_duplicate_and_already_done_replies_keep_state_updates(self):
        task, request_id = await self.start_request()
        await self.reply(request_id)
        self.assertEqual(await asyncio.wait_for(task, 1), {"status": "SUCCESS"})
        await self.reply(request_id)
        for cancelled in (False, True):
            waiter = asyncio.get_running_loop().create_future()
            if cancelled:
                waiter.cancel()
            else:
                waiter.set_result({"status": "SUCCESS"})
            self.glagol.waiters[request_id] = waiter
            await self.reply(request_id)
            self.glagol.waiters.pop(request_id)
        await self.assert_next_request_works()

    async def test_send_error_cleans_waiter(self):
        self.ws.send_error = ConnectionResetError("fixture send error")
        with self.assertLogs("glagol_test_core.yandex_glagol", level="ERROR"):
            result = await self.glagol.send({"command": "fixture-error"})
        self.assertIn("error", result)
        self.assertEqual(self.glagol.waiters, {})
        self.ws.send_error = None
        await self.assert_next_request_works()

    async def test_response_formats_keep_existing_results(self):
        cases = [
            ({"response": {"card": {"text": "card"}}}, {"text": "card"}),
            (
                {
                    "payload": {
                        "response": {"cards": [{"text": "first"}, {"text": "second"}]}
                    }
                },
                {"text": "first"},
            ),
            ({"response": {"is_streaming": True}}, {"is_streaming": True}),
            ({"response": {"output_speech": {"text": "speech"}}}, {"text": "speech"}),
            (
                {"payload": {}, "response": {"card": {"text": "direct"}}},
                {"text": "direct"},
            ),
            ({"response": {}}, {}),
            ({"payload": {"missing": "response"}}, {}),
        ]
        for response, fields in cases:
            with self.subTest(response=response):
                task, request_id = await self.start_request()
                await self.reply(request_id, response)
                self.assertEqual(
                    await asyncio.wait_for(task, 1), {"status": "SUCCESS", **fields}
                )
                self.assertNotIn(request_id, self.glagol.waiters)

    async def test_instances_own_separate_waiters(self):
        other = YandexGlagol(self.session, {"name": "Other regression speaker"})
        self.assertIsNot(self.glagol.waiters, other.waiters)


if __name__ == "__main__":
    unittest.main()
