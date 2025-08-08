import asyncio
import threading
from pathlib import Path

import numpy as np

from aiogram import Bot
from aiogram.types import FSInputFile

from data_classes.frame import FrameData
from visualization.video_writer import VideoWriter

class Notify:
    def __init__(self, config, project_dir: Path):
        self.config = config
        self.project_dir = project_dir

        self.token = config["notify"]["token"]
        self.chat_id = config["notify"]["chat_id"]

        self.buffer_size = config["notify"]["buffer_size"]
        output_path = Path(config["notify"]["output_path"]).expanduser()
        self.output_path = output_path if output_path.is_absolute() else project_dir / output_path
        Path.mkdir(self.output_path, parents=True, exist_ok=True)

        self.location = config["notify"]["location"]
        self.location_name = config["notify"]["location_name"]

        self.bot = Bot(token=self.token)

        self.buffer: list[np.ndarray] = []

        self.frame_id = 0
        self.video_id = 0
        self.sending = False

    def process(self, frame_data: FrameData) -> None:
        self.frame_id += 1

        self.buffer.append(frame_data.frame_out)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        for person in frame_data.people.values():
            if person.crash:
                self.send_notification()
                self.sending = True

    async def send_message(self) -> None:
        await self.bot.send_message(
            self.chat_id,
            f"⚠️ Detected accident\n📌{self.location_name}\n📍 {self.location[0]}, {self.location[1]}",
        )
        await self.bot.send_location(
            self.chat_id,
            self.location[0],
            self.location[1],
        )

    async def send_video(self) -> None:
        self.video_id += 1
        file_name = f"a_{self.video_id}.mkv"
        file_path = self.output_path / file_name

        writer = VideoWriter(self.config, self.project_dir, filepath=str(file_path))
        for frame in self.buffer.copy():
            writer.process(frame)

        writer.close_current_writer()
        file = FSInputFile(file_path)
        try:
            await self.bot.send_video(self.chat_id, file)
        except Exception as e:
            print(f"Error send video {e}")


    def send_notification(self) -> None:
        if self.sending:
            return

        self.sending = True
        start_frame_id = self.frame_id

        t = threading.Thread(
            target=self._notification_thread_entry,
            args=(start_frame_id,),
            daemon=True,
        )
        t.start()

    def _notification_thread_entry(self, start_frame_id: int) -> None:
        asyncio.run(self._notification_worker(start_frame_id))

    async def _notification_worker(self, start_frame_id: int) -> None:
        try:
            await self.send_message()

            half = max(1, self.buffer_size // 2)
            target_frame_id = start_frame_id + half
            while self.frame_id < target_frame_id:
                await asyncio.sleep(0.02)

            await self.send_video()

        except Exception as e:
            print(e)
        finally:
            pass
            # self.sending = False