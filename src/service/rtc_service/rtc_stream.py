import asyncio
import json
import time
import uuid
import weakref
from typing import Optional, Dict

import numpy as np
# noinspection PyPackageRequirements
from fastrtc import AsyncAudioVideoStreamHandler, AudioEmitType, VideoEmitType
from loguru import logger

from chat_engine.common.client_handler_base import ClientHandlerDelegate, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from engine_utils.interval_counter import IntervalCounter

def _get_h264_encoder_info():
    """Get H.264 encoder info dynamically to avoid circular imports"""
    try:
        from handlers.client.rtc_client import client_handler_rtc
        return client_handler_rtc._selected_h264_encoder, client_handler_rtc._actual_h264_encoder
    except Exception:
        return "unknown", None


class RtcStream(AsyncAudioVideoStreamHandler):
    def __init__(self,
                 session_id: Optional[str],
                 expected_layout="mono",
                 input_sample_rate=16000,
                 output_sample_rate=24000,
                 output_frame_size=480,
                 fps=30,
                 stream_start_delay = 0.5,
                 ):
        super().__init__(
            expected_layout=expected_layout,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            fps=fps
        )
        self.client_handler_delegate: Optional[ClientHandlerDelegate] = None
        self.client_session_delegate: Optional[ClientSessionDelegate] = None

        self.weak_factory: Optional[weakref.ReferenceType[RtcStream]] = None

        self.session_id = session_id
        self.stream_start_delay = stream_start_delay

        self.chat_channel = None
        self.first_audio_emitted = False

        self.quit = asyncio.Event()
        self.last_frame_time = 0

        self.emit_counter = IntervalCounter("emit counter")

        self.start_time = None
        self.timestamp_base = self.input_sample_rate

        self.streams: Dict[str, RtcStream] = {}


    # copy is used as create_instance in fastrtc
    def copy(self, **kwargs) -> AsyncAudioVideoStreamHandler:
        try:
            if self.client_handler_delegate is None:
                raise Exception("ClientHandlerDelegate is not set.")
            session_id = kwargs.get("webrtc_id", None)
            if session_id is None:
                session_id = uuid.uuid4().hex
            
            # Log codec information
            selected_encoder, _ = _get_h264_encoder_info()
            logger.debug(f"[{session_id}] H.264 encoder: {selected_encoder}")
            
            new_stream = RtcStream(
                session_id,
                expected_layout=self.expected_layout,
                input_sample_rate=self.input_sample_rate,
                output_sample_rate=self.output_sample_rate,
                output_frame_size=self.output_frame_size,
                fps=self.fps,
                stream_start_delay=self.stream_start_delay,
            )
            new_stream.weak_factory = weakref.ref(self)
            new_session_delegate = self.client_handler_delegate.start_session(
                session_id=session_id,
                timestamp_base=self.input_sample_rate,
            )
            new_stream.client_session_delegate = new_session_delegate
            if session_id in self.streams:
                msg = f"Stream {session_id} already exists."
                raise RuntimeError(msg)
            self.streams[session_id] = new_stream
            return new_stream
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to create stream: {e}")
            raise

    async def emit(self) -> AudioEmitType:
        try:
            if not self.first_audio_emitted:
                self.first_audio_emitted = True
                logger.debug(f"[RTC Stream] First audio emit started for session {self.session_id}")

            while not self.quit.is_set():
                chat_data = await self.client_session_delegate.get_data(EngineChannelType.AUDIO)
                if chat_data is None or chat_data.data is None:
                    continue
                audio_array = chat_data.data.get_main_data()
                if audio_array is None:
                    continue

                # Ensure audio is 1D float32 array in [-1, 1] range
                audio_array = audio_array.squeeze().astype(np.float32)

                # Log audio info for debugging
                logger.info(f"[RTC Stream] Emitting audio: samples={len(audio_array)}, dtype={audio_array.dtype}, min={audio_array.min():.3f}, max={audio_array.max():.3f}")

                self.emit_counter.add_property("audio_emit", len(audio_array) / self.output_sample_rate)
                return self.output_sample_rate, audio_array

        except Exception as e:
            logger.opt(exception=e).error("Error in emit: ")
            raise

    async def video_emit(self) -> VideoEmitType:
        try:
            if not self.first_audio_emitted:
                await asyncio.sleep(0.1)

            # Log actual encoder being used (for verification)
            selected_encoder, actual_encoder = _get_h264_encoder_info()
            if actual_encoder:
                logger.debug(f"[{self.session_id}] Using H.264 encoder: {actual_encoder}")

            self.emit_counter.add_property("video_emit")

            # Track consecutive empty frames for placeholder logic
            empty_frame_count = 0
            max_empty_frames = 3  # After 3 empty frames, return placeholder

            while not self.quit.is_set():
                get_data_start = time.perf_counter()
                video_frame_data: ChatData = await self.client_session_delegate.get_data(EngineChannelType.VIDEO)
                get_data_wait_time = time.perf_counter() - get_data_start

                # Log slow data retrieval
                if get_data_wait_time > 0.05:
                    logger.debug(f"[{self.session_id}] Slow video data retrieval: {get_data_wait_time:.3f}s")

                if video_frame_data is None or video_frame_data.data is None:
                    empty_frame_count += 1
                    # Return a black placeholder frame to keep WebRTC happy
                    # Use 320x240 which is compatible with H264 encoder
                    if empty_frame_count >= max_empty_frames:
                        empty_frame_count = 0
                        return np.zeros((240, 320, 3), dtype=np.uint8)
                    continue

                frame_data = video_frame_data.data.get_main_data().squeeze()
                if frame_data is None:
                    empty_frame_count += 1
                    if empty_frame_count >= max_empty_frames:
                        empty_frame_count = 0
                        return np.zeros((240, 320, 3), dtype=np.uint8)
                    continue

                empty_frame_count = 0
                return frame_data
        except Exception as e:
            logger.opt(exception=e).error("Error in video_emit")
            raise

    async def receive(self, frame: tuple[int, np.ndarray]):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        _, array = frame
        self.client_session_delegate.put_data(
            EngineChannelType.AUDIO,
            array,
            timestamp,
            self.input_sample_rate,
        )

    async def video_receive(self, frame):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        self.client_session_delegate.put_data(
            EngineChannelType.VIDEO,
            frame,
            timestamp,
            self.fps,
        )

    def set_channel(self, channel):
            super().set_channel(channel)
            self.chat_channel = channel

            # 设置调试日志回调函数，让后端能够发送日志到前端
            def send_debug_log(log_type: str, message: str, image_base64: str = None):
                """发送调试日志到前端
                Args:
                    log_type: 日志类型，如 'info', 'video_analysis', 'llm_request' 等
                    message: 日志消息
                    image_base64: 可选的base64编码的图片缩略图
                """
                try:
                    log_data = {
                        'type': 'debug_log',
                        'log_type': log_type,
                        'message': message,
                        'timestamp': time.time()
                    }
                    if image_base64:
                        log_data['image'] = image_base64
                    self.chat_channel.send(json.dumps(log_data))
                except Exception as e:
                    logger.error(f"Failed to send debug log: {e}")

            # 将回调函数设置到 shared_states
            if self.client_session_delegate and self.client_session_delegate.shared_states:
                self.client_session_delegate.shared_states.debug_log_callback = send_debug_log

            async def process_chat_history():
                role = None
                chat_id = None
                last_speech_id = None
                while not self.quit.is_set():
                    chat_data = await self.client_session_delegate.get_data(EngineChannelType.TEXT)
                    if chat_data is None or chat_data.data is None:
                        continue
                    logger.debug(f"Got chat data {str(chat_data)}")
                    current_role = 'human' if chat_data.type == ChatDataType.HUMAN_TEXT else 'avatar'
                    current_speech_id = chat_data.data.get_meta("speech_id")
                    # 当 role 改变或 speech_id 改变时，生成新的 chat_id
                    if current_role != role or (current_speech_id and current_speech_id != last_speech_id):
                        chat_id = uuid.uuid4().hex
                    role = current_role
                    last_speech_id = current_speech_id
                    self.chat_channel.send(json.dumps({'type': 'chat', 'message': chat_data.data.get_main_data(),
                                                        'id': chat_id, 'role': current_role}))
            asyncio.create_task(process_chat_history())
                
            @channel.on("message")
            def _(message):
                logger.info(f"Received message Custom: {message}")
                try:
                    message = json.loads(message)
                except Exception as e:
                    logger.info(e)
                    message = {}

                if self.client_session_delegate is None:
                    logger.warning(f"[RTC Stream] Message received but client_session_delegate is None: {message}")
                    return
                timestamp = self.client_session_delegate.get_timestamp()
                if timestamp[0] / timestamp[1] < self.stream_start_delay:
                    return
                logger.info(f'on_chat_datachannel: {message}')

                msg_type = message.get('type', '')
                if msg_type == 'stop_chat':
                    self.client_session_delegate.emit_signal(
                        ChatSignal(
                            type=ChatSignalType.INTERRUPT,
                            source_type=ChatSignalSourceType.CLIENT,
                            source_name="rtc",
                        )
                    )
                elif msg_type == 'video_source_changed':
                    # 当用户切换视频源（摄像头↔屏幕共享）时，立即触发视频分析
                    source = message.get('source', 'unknown')
                    logger.info(f"[RTC Stream] Video source changed to: {source}")
                    if self.client_session_delegate.shared_states:
                        self.client_session_delegate.shared_states.trigger_immediate_analysis = True
                        logger.info("[RTC Stream] Triggered immediate video analysis")
                    else:
                        logger.warning("[RTC Stream] shared_states is None, cannot trigger immediate analysis")
                elif msg_type == 'chat':
                    channel.send(json.dumps({'type': 'avatar_end'}))
                    if self.client_session_delegate.shared_states.enable_vad is False:
                        return
                    self.client_session_delegate.shared_states.enable_vad = False
                    self.client_session_delegate.emit_signal(
                        ChatSignal(
                            # begin a new round of responding
                            type=ChatSignalType.BEGIN,
                            stream_type=ChatDataType.AVATAR_AUDIO,
                            source_type=ChatSignalSourceType.CLIENT,
                            source_name="rtc",
                        )
                    )
                    self.client_session_delegate.put_data(
                        EngineChannelType.TEXT,
                        message['data'],
                        loopback=True
                    )
                # else:

                # channel.send(json.dumps({"type": "chat", "unique_id": unique_id, "message": message}))
          
    async def on_chat_datachannel(self, message: Dict, channel):
        # {"type":"chat",id:"标识属于同一段话", "message":"Hello, world!"}
        # unique_id = uuid.uuid4().hex
        pass
    def shutdown(self):
        self.quit.set()
        factory = None
        if self.weak_factory is not None:
            factory = self.weak_factory()
        if factory is None:
            factory = self
        self.client_session_delegate = None
        if factory.client_handler_delegate is not None:
            factory.client_handler_delegate.stop_session(self.session_id)
