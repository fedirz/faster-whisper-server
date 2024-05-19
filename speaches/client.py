# TODO: move out of `speaches` package
import asyncio
import signal

import httpx
from httpx_ws import AsyncWebSocketSession, WebSocketDisconnect, aconnect_ws
from wsproto.connection import ConnectionState

CHUNK = 1024 * 4
AUDIO_RECORD_CMD = "arecord -D default -f S16_LE -r 16000 -c 1 -t raw"
COPY_TO_CLIPBOARD_CMD = "wl-copy"
NOTIFY_CMD = "notify-desktop"

client = httpx.AsyncClient(base_url="ws://localhost:8000")


async def audio_sender(ws: AsyncWebSocketSession) -> None:
    process = await asyncio.create_subprocess_shell(
        AUDIO_RECORD_CMD,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        while not process.stdout.at_eof():
            data = await process.stdout.read(CHUNK)
            if ws.connection.state != ConnectionState.OPEN:
                break
            await ws.send_bytes(data)
    except Exception as e:
        print(e)
    finally:
        process.kill()


async def transcription_receiver(ws: AsyncWebSocketSession) -> None:
    transcription = ""
    notification_id: int | None = None
    try:
        while True:
            data = await ws.receive_text()
            if not data:
                break
            transcription += data
            await copy_to_clipboard(transcription)
            notification_id = await notify(transcription, replaces_id=notification_id)
    except WebSocketDisconnect:
        pass
    print(transcription)


async def copy_to_clipboard(text: str) -> None:
    process = await asyncio.create_subprocess_shell(
        COPY_TO_CLIPBOARD_CMD, stdin=asyncio.subprocess.PIPE
    )
    await process.communicate(input=text.encode("utf-8"))
    await process.wait()


async def notify(text: str, replaces_id: int | None = None) -> int:
    cmd = ["notify-desktop", "--app-name", "Speaches"]
    if replaces_id is not None:
        cmd.extend(["--replaces-id", str(replaces_id)])
    cmd.append("'Speaches'")
    cmd.append(f"'{text}'")
    process = await asyncio.create_subprocess_shell(
        " ".join(cmd),
        stdout=asyncio.subprocess.PIPE,
    )
    await process.wait()
    assert process.stdout is not None
    notification_id = (await process.stdout.read()).decode("utf-8")
    return int(notification_id)


async def main() -> None:
    async with aconnect_ws("/v1/audio/transcriptions", client) as ws:
        async with asyncio.TaskGroup() as tg:
            sender_task = tg.create_task(audio_sender(ws))
            receiver_task = tg.create_task(transcription_receiver(ws))

            async def on_interrupt():
                sender_task.cancel()
                receiver_task.cancel()
                await asyncio.gather(sender_task, receiver_task)

            asyncio.get_running_loop().add_signal_handler(
                signal.SIGINT,
                lambda: asyncio.create_task(on_interrupt()),
            )


asyncio.run(main())
# poetry --directory /home/nixos/code/speaches run python /home/nixos/code/speaches/speaches/client.py
