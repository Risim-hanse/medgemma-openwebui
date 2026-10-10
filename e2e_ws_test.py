"""End-to-end test of the Chainlit UI over its real socket.io transport.

STATUS: partially verified -- see note below.

Chainlit mounts engine.io at /ws/socket.io (NOT /ws/chat). Connecting
works, but Chainlit's socket.connect handler does `auth.get("threadId")`,
and python-socketio 5.17 delivers `auth` as None in this headless setup,
so the handshake handler raises before our chat handlers are reached. A real
browser sends a proper CONNECT auth payload, so this is a harness gap, not
an app defect.

What IS verified directly (see /tmp streaming check and smoke_test.py):
  - model loads on the T4 in bfloat16
  - TextIteratorStreamer + threading produces live token chunks
  - content is clinically correct

If you need this green, drive it through a headless browser instead.
"""

import asyncio
import sys

import socketio

URL = "http://127.0.0.1:8001"
# NB: Chainlit mounts engine.io at /ws/socket.io (chainlit/server.py),
# not /ws/chat.
CHAT_PATH = "/ws/socket.io"

seen = []
chunks = []


async def main():
    sio = socketio.AsyncClient()

    @sio.on("connect")
    async def on_connect():
        print("[ws] connected")

    @sio.on("task_start")
    async def on_task_start(data=None):
        print("[event] task_start -> model loading / generating")

    @sio.on("message_start")
    async def on_message_start(data=None):
        print("[event] message_start")

    @sio.on("message_chunk")
    async def on_chunk(data=None):
        d = data.get("delta") or data.get("content") or "" if isinstance(data, dict) else ""
        if d.strip():
            chunks.append(d)
            if len(chunks) <= 6:
                print(f"[stream] {d.strip()[:90]}")

    @sio.on("message_end")
    async def on_end(data=None):
        print("[event] message_end")

    @sio.on("task_end")
    async def on_task_end(data=None):
        print("[event] task_end")

    @sio.on("error")
    async def on_error(data=None):
        print("[ERROR]", str(data)[:400])

    @sio.on("*")
    async def catch_all(event, data=None):
        seen.append(str(event))

    # Chainlit's socket.connect handler does auth.get("threadId"), so `auth`
    # must be a dict -- passing auth=None crashes the handler.
    await sio.connect(
        URL,
        socketio_path=CHAT_PATH,
        transports=["websocket"],
        auth={},
        headers={"Origin": "http://127.0.0.1:8001"},
    )
    print("[ws] connected, sid =", sio.sid)
    await sio.emit("connect", {"userInfo": {"identifier": "smoke", "metadata": {}}})
    await asyncio.sleep(3)

    print("[ws] sending user message...")
    await sio.emit(
        "client_message",
        {
            "name": "User",
            "content": "What is the capital of France? One word only.",
            "elements": [],
        },
    )

    # Wait for the turn to finish.
    for _ in range(180):
        await asyncio.sleep(2)
        if "task_end" in seen or "message_end" in seen:
            break

    full = "".join(chunks)
    print(f"\nstreamed {len(chunks)} chunks, {len(full)} chars")
    print("streamed text:", repr(full[:200]))

    await sio.disconnect()

    ok = len(chunks) > 0 and full.strip() != ""
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))