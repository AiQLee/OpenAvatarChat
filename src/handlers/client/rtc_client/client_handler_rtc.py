import asyncio
import base64
import io
from pathlib import Path
from typing import Dict, Optional, cast, Union, Tuple
from uuid import uuid4

from loguru import logger

from engine_utils.directory_info import DirectoryInfo
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import glob as glob_module
import gradio
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel

# ============================================================================
# H.264 Hardware Encoder Configuration (must execute before importing fastrtc)
# ============================================================================
from aiortc.codecs import CODECS, h264
from aiortc import codecs as aiortc_codecs
import av
import fractions

# Global variables for H.264 encoder selection
_selected_h264_encoder = 'libx264'
_actual_h264_encoder = None
_AVError = getattr(av, "AVError", av.error.ExternalError)
_AVCodecError = getattr(av.error, "Error", av.error.ExternalError)

def _prioritize_h264():
    """Prioritize H.264 codec over VP8 in aiortc codec list"""
    video_codecs = CODECS["video"]
    h264_codecs = [c for c in video_codecs if "H264" in c.mimeType]
    other_codecs = [c for c in video_codecs if "H264" not in c.mimeType]
    CODECS["video"] = h264_codecs + other_codecs
    logger.info(f"Video codec priority: {[c.mimeType for c in CODECS['video'][:3]]}")

def _configure_h264_hardware_encoding():
    """Configure H.264 to use hardware encoder (must execute before importing fastrtc)"""
    global _selected_h264_encoder, _actual_h264_encoder
    
    # Configure H.264 bitrate
    h264.DEFAULT_BITRATE = 1500000  # 1.5 Mbps default
    h264.MIN_BITRATE = 500000       # 0.5 Mbps minimum
    h264.MAX_BITRATE = 2500000      # 2.5 Mbps maximum
    
    # Detect available hardware encoders (priority: NVENC > QSV > VideoToolbox)
    hardware_encoders = ['h264_nvenc', 'h264_qsv', 'h264_videotoolbox']
    for encoder in hardware_encoders:
        try:
            test_codec = av.CodecContext.create(encoder, "w")
            test_codec.width = 640
            test_codec.height = 480
            test_codec.pix_fmt = "yuv420p"
            test_codec.framerate = fractions.Fraction(30, 1)
            test_codec.time_base = fractions.Fraction(1, 30)
            _selected_h264_encoder = encoder
            logger.info(f"Detected H.264 hardware encoder: {encoder}")
            break
        except Exception as e:
            logger.debug(f"Hardware encoder {encoder} not available: {e}")
    
    if _selected_h264_encoder == 'libx264':
        logger.warning("No hardware encoder available, will use libx264 (CPU encoding)")
    
    # Monkey patch H264Encoder._encode_frame to use hardware encoder with fallback
    def patched_encode_frame(self, frame, force_keyframe):
        global _selected_h264_encoder, _actual_h264_encoder
        if self.codec and (
            frame.width != self.codec.width or frame.height != self.codec.height
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        else:
            frame.pict_type = av.video.frame.PictureType.NONE

        encoder_to_use = getattr(self, "_preferred_encoder", _selected_h264_encoder)
        fallback_attempted = False
        data_to_send = b""

        while True:
            if self.codec is None:
                # Try to create encoder, fallback to software encoder if hardware fails
                codec_created = False
                
                try:
                    self.codec = av.CodecContext.create(encoder_to_use, "w")
                    self.codec.width = frame.width
                    self.codec.height = frame.height
                    self.codec.bit_rate = self.target_bitrate
                    self.codec.pix_fmt = "yuv420p"
                    self.codec.framerate = fractions.Fraction(h264.MAX_FRAME_RATE, 1)
                    self.codec.time_base = fractions.Fraction(1, h264.MAX_FRAME_RATE)
                    
                    # Configure encoder-specific options
                    if encoder_to_use == 'libx264':
                        self.codec.options = {"level": "31", "tune": "zerolatency"}
                        self.codec.profile = "Baseline"
                    elif encoder_to_use == 'h264_nvenc':
                        self.codec.options = {"preset": "llhq", "zerolatency": "1", "profile": "baseline"}
                    elif encoder_to_use == 'h264_qsv':
                        self.codec.options = {"preset": "veryfast", "profile": "baseline"}
                    elif encoder_to_use == 'h264_videotoolbox':
                        self.codec.options = {"realtime": "1", "profile": "baseline"}
                    
                    codec_created = True
                    logger.info(f"H.264 encoder created: {encoder_to_use}")
                    
                except Exception as e:
                    logger.warning(f"Failed to create {encoder_to_use} encoder: {e}")
                    
                    # Fallback to libx264 if hardware encoder fails
                    if encoder_to_use != 'libx264':
                        logger.info("Falling back to libx264 software encoder")
                        try:
                            self.codec = av.CodecContext.create("libx264", "w")
                            self.codec.width = frame.width
                            self.codec.height = frame.height
                            self.codec.bit_rate = self.target_bitrate
                            self.codec.pix_fmt = "yuv420p"
                            self.codec.framerate = fractions.Fraction(h264.MAX_FRAME_RATE, 1)
                            self.codec.time_base = fractions.Fraction(1, h264.MAX_FRAME_RATE)
                            self.codec.options = {"level": "31", "tune": "zerolatency"}
                            self.codec.profile = "Baseline"
                            
                            encoder_to_use = "libx264"
                            codec_created = True
                            logger.info("H.264 encoder created: libx264 (fallback)")
                            
                        except Exception as fallback_error:
                            logger.error(f"Failed to create fallback encoder: {fallback_error}")
                            raise
                    else:
                        logger.error(f"Failed to create libx264 encoder: {e}")
                        raise
                
                if codec_created:
                    actual_encoder = self.codec.name
                    _actual_h264_encoder = actual_encoder
                    self._preferred_encoder = encoder_to_use

            try:
                data_to_send = b""
                for package in self.codec.encode(frame):
                    data_to_send += bytes(package)
                break
            except (_AVError, _AVCodecError) as encode_error:
                if fallback_attempted or encoder_to_use == 'libx264':
                    logger.error(f"H.264 encode failed using {encoder_to_use}: {encode_error}")
                    raise
                
                fallback_attempted = True
                logger.warning(
                    f"H.264 encode failed using {encoder_to_use}, switching to libx264: {encode_error}"
                )
                if self.codec is not None:
                    try:
                        self.codec.close()
                    except Exception as close_error:
                        logger.debug(f"Error closing codec during fallback: {close_error}")
                self.codec = None
                self.buffer_data = b""
                self.buffer_pts = None
                encoder_to_use = 'libx264'
                self._preferred_encoder = encoder_to_use
                _selected_h264_encoder = 'libx264'
                force_keyframe = True
                frame.pict_type = av.video.frame.PictureType.I
                continue

        if data_to_send:
            yield from self._split_bitstream(data_to_send)
    
    # Patch get_encoder
    original_get_encoder = aiortc_codecs.get_encoder
    def patched_get_encoder(codec):
        encoder = original_get_encoder(codec)
        logger.debug(f"get_encoder({codec.mimeType}) -> {type(encoder).__name__}")
        return encoder
    
    # Patch RTCPeerConnection.setRemoteDescription to enforce H.264 codec priority
    # This ensures H.264 is selected during WebRTC negotiation instead of VP8
    from aiortc import RTCPeerConnection
    original_set_remote = RTCPeerConnection.setRemoteDescription
    
    async def patched_set_remote_description(self, sessionDescription):
        """Re-order transceiver codecs after SDP negotiation to prioritize H.264"""
        logger.debug(f"setRemoteDescription called with {sessionDescription.type}")
        
        # Call original method (creates/configures transceivers and negotiates codecs)
        await original_set_remote(self, sessionDescription)
        
        # Re-order video transceiver codecs to prioritize H.264
        from aiortc.codecs import get_capabilities
        from aiortc.rtcpeerconnection import filter_preferred_codecs
        
        for transceiver in self._RTCPeerConnection__transceivers:
            if transceiver.kind == "video":
                logger.debug(f"Transceiver codecs before: {[c.mimeType for c in transceiver._codecs[:3]]}")
                
                # Get our re-ordered codec capabilities (H.264 first)
                capabilities = get_capabilities("video")
                current_codecs = transceiver._codecs
                
                # Manually call filter_preferred_codecs to re-order
                refiltered = filter_preferred_codecs(current_codecs, capabilities.codecs)
                transceiver._codecs = refiltered
                
                logger.info(f"Video codecs negotiated: {[c.mimeType for c in transceiver._codecs[:2]]}")
    
    # Apply all patches
    h264.H264Encoder._encode_frame = patched_encode_frame
    aiortc_codecs.get_encoder = patched_get_encoder
    RTCPeerConnection.setRemoteDescription = patched_set_remote_description
    logger.info("H.264 encoder configuration completed")

# Execute configuration before importing fastrtc
_prioritize_h264()
_configure_h264_hardware_encoding()

# Import fastrtc after H.264 configuration
# noinspection PyPackageRequirements
from fastrtc import Stream  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402
from chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate  # noqa: E402
from chat_engine.common.engine_channel_type import EngineChannelType  # noqa: E402
from chat_engine.common.handler_base import HandlerDataInfo, HandlerDetail, HandlerBaseInfo  # noqa: E402
from chat_engine.contexts.handler_context import HandlerContext  # noqa: E402
from chat_engine.contexts.session_context import SessionContext  # noqa: E402
from chat_engine.data_models.chat_data.chat_data_model import ChatData  # noqa: E402
from chat_engine.data_models.chat_data_type import ChatDataType  # noqa: E402
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel  # noqa: E402
from chat_engine.data_models.chat_signal import ChatSignal  # noqa: E402
from chat_engine.data_models.runtime_data.data_bundle import (  # noqa: E402
    DataBundleDefinition, DataBundleEntry, VariableSize, DataBundle  # noqa: E402
)  # noqa: E402
from service.rtc_service.rtc_provider import RTCProvider  # noqa: E402
from service.rtc_service.rtc_stream import RtcStream  # noqa: E402


class RtcClientSessionDelegate(ClientSessionDelegate):
    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        self._event_loop = None  # Will be set when get_data is first called
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.VIDEO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
        }
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }

    def _put_to_queue_threadsafe(self, queue: asyncio.Queue, data):
        """Thread-safe method to put data into an asyncio queue from a non-async thread."""
        if self._event_loop is None:
            # Fallback: try to use put_nowait directly (may not be thread-safe)
            logger.warning("[RtcClientSessionDelegate] Event loop not set, using put_nowait directly")
            queue.put_nowait(data)
        else:
            # Use call_soon_threadsafe to safely put data from another thread
            self._event_loop.call_soon_threadsafe(queue.put_nowait, data)

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        # Capture the event loop on first call (this is called from async context)
        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()
            logger.debug(f"[RtcClientSessionDelegate] Captured event loop for thread-safe queue access")

        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None
        if timeout is not None and timeout > 0:
            try:
                data = await asyncio.wait_for(data_queue.get(), timeout)
            except asyncio.TimeoutError:
                return None
        else:
            data = await data_queue.get()
        return data

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None, samplerate: Optional[int] = None, loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        data_bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            data_bundle.add_meta('human_text_end', True)
            data_bundle.add_meta('speech_id', str(uuid4()))
            data_bundle.set_main_data(data)
        else:
            return
        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        self.data_submitter.submit(chat_data)
        if loopback:
            self.output_queues[modality].put_nowait(chat_data)

    def get_timestamp(self):
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        pass

    def clear_data(self):
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()


class ClientRtcConfigModel(HandlerBaseConfigModel, BaseModel):
    connection_ttl: int = Field(default=900)
    turn_config: Optional[Dict] = Field(default=None)


class ClientRtcContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ClientRtcConfigModel] = None
        self.client_session_delegate: Optional[RtcClientSessionDelegate] = None


class ClientHandlerRtc(ClientHandlerBase):
    def __init__(self):
        super().__init__()
        self.engine_config = None
        self.handler_config = None
        self.rtc_streamer_factory: Optional[RtcStream] = None

        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ClientRtcConfigModel,
            client_session_delegate_class=RtcClientSessionDelegate,
        )

    def prepare_rtc_definitions(self):
        self.rtc_streamer_factory = RtcStream(
            session_id=None,
            expected_layout="mono",
            input_sample_rate=16000,
            output_sample_rate=24000,
            output_frame_size=480,
            fps=30,
            stream_start_delay=0.5,
        )
        self.rtc_streamer_factory.client_handler_delegate = self.handler_delegate

        audio_output_definition = DataBundleDefinition()
        audio_output_definition.add_entry(DataBundleEntry.create_audio_entry(
            "mic_audio",
            1,
            16000,
        ))
        audio_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.AUDIO] = audio_output_definition

        video_output_definition = DataBundleDefinition()
        video_output_definition.add_entry(DataBundleEntry.create_framed_entry(
            "camera_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],
            0,
            30
        ))
        video_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.VIDEO] = video_output_definition

        text_output_definition = DataBundleDefinition()
        text_output_definition.add_entry(DataBundleEntry.create_text_entry(
            "human_text",
        ))
        text_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.TEXT] = text_output_definition

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        self.engine_config = engine_config
        self.handler_config = cast(ClientRtcConfigModel, handler_config)
        self.prepare_rtc_definitions()

    def setup_rtc_ui(self, ui, parent_block, fastapi: FastAPI, avatar_config):
        turn_entity = RTCProvider().prepare_rtc_configuration(self.handler_config.turn_config)
        if turn_entity is None:
            turn_entity = RTCProvider().prepare_rtc_configuration(self.engine_config.turn_config)

        webrtc = Stream(
            modality="audio-video",
            mode="send-receive",
            time_limit=self.handler_config.connection_ttl,
            rtc_configuration=turn_entity.rtc_configuration if turn_entity is not None else None,
            handler=self.rtc_streamer_factory,
            concurrency_limit=self.handler_config.concurrent_limit,
        )
        webrtc.mount(fastapi)

        @fastapi.get('/openavatarchat/initconfig')
        async def init_config():
            config = {
                "avatar_config": avatar_config,
                "rtc_configuration": turn_entity.rtc_configuration if turn_entity is not None else None,
            }
            return JSONResponse(status_code=200, content=config)

        # TTS测试API请求模型
        class TTSTestRequest(PydanticBaseModel):
            text: str

        @fastapi.post('/tts/generate')
        async def tts_generate(request: TTSTestRequest):
            """TTS测试端点 - 使用火山引擎OpenSpeech TTS生成语音"""
            try:
                import requests as http_requests
                import soundfile as sf
                import uuid as uuid_module
                import json

                text = request.text.strip()
                if not text:
                    return JSONResponse(status_code=400, content={"error": "文本不能为空"})

                logger.info(f"[TTS Test] Generating audio for: {text[:50]}...")

                # 火山引擎OpenSpeech TTS V3 API配置 - 基于官方HTTP示例
                appid = "9825753006"
                access_token = "PPAbNsXfUaOrb-zZkk2rMaFqM9vdjOZy"
                resource_id = "volc.service_type.10029"  # 大模型语音合成资源ID
                voice_type = "zh_female_cancan_mars_bigtts"  # 灿灿大模型音色
                sample_rate = 24000

                # V3 API URL (HTTP单向流式)
                api_url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

                # V3 API Headers格式
                headers = {
                    "X-Api-App-Id": appid,
                    "X-Api-Access-Key": access_token,
                    "X-Api-Resource-Id": resource_id,
                    "Content-Type": "application/json",
                    "Connection": "keep-alive"
                }

                # V3 API 请求体格式
                request_body = {
                    "user": {
                        "uid": "tts_test_user"
                    },
                    "req_params": {
                        "text": text,
                        "speaker": voice_type,
                        "audio_params": {
                            "format": "mp3",
                            "sample_rate": sample_rate,
                            "enable_timestamp": False
                        },
                        "additions": "{\"explicit_language\":\"zh\",\"disable_markdown_filter\":true}"
                    }
                }

                logger.info(f"[TTS Test] Calling OpenSpeech V3 API...")
                logger.info(f"[TTS Test] URL: {api_url}")
                logger.info(f"[TTS Test] Headers: {headers}")

                # 使用流式响应
                response = http_requests.post(api_url, headers=headers, json=request_body, timeout=30, stream=True)

                logger.info(f"[TTS Test] API response status: {response.status_code}")
                logger.info(f"[TTS Test] Response headers: {dict(response.headers)}")

                if response.status_code != 200:
                    error_text = response.text[:500] if hasattr(response, 'text') else str(response.content[:500])
                    logger.error(f"[TTS Test] API error response: {error_text}")
                    return JSONResponse(status_code=500, content={
                        "error": f"TTS API错误: {response.status_code} - {error_text}"
                    })

                # 解析流式响应，收集所有音频数据
                audio_data = bytearray()
                line_count = 0
                for line in response.iter_lines(decode_unicode=True):
                    line_count += 1
                    if not line:
                        continue
                    try:
                        logger.debug(f"[TTS Test] Received line {line_count}: {line[:200]}...")
                        data = json.loads(line)
                        code = data.get("code", 0)

                        # 正常音频数据
                        if code == 0 and "data" in data and data["data"]:
                            chunk_audio = base64.b64decode(data["data"])
                            audio_data.extend(chunk_audio)
                            logger.info(f"[TTS Test] Received audio chunk: {len(chunk_audio)} bytes")
                            continue

                        # 结束标记
                        if code == 20000000:
                            logger.info(f"[TTS Test] TTS completed, usage: {data.get('usage', {})}")
                            break

                        # 错误
                        if code > 0 and code != 20000000:
                            logger.error(f"[TTS Test] API error in stream: {data}")
                            return JSONResponse(status_code=500, content={
                                "error": f"TTS流式响应错误: code={code}, message={data.get('message', 'Unknown error')}"
                            })
                    except json.JSONDecodeError as e:
                        logger.warning(f"[TTS Test] Failed to parse line {line_count}: {line[:100]}, error: {e}")

                logger.info(f"[TTS Test] Total lines processed: {line_count}")

                if not audio_data:
                    return JSONResponse(status_code=500, content={"error": "没有音频数据返回"})

                audio_bytes = bytes(audio_data)
                logger.info(f"[TTS Test] Audio collected, size={len(audio_bytes)} bytes")

                # MP3格式需要转换为WAV
                import librosa
                audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate)

                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio_array, sr, format='WAV')
                wav_buffer.seek(0)
                wav_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')

                logger.info(f"[TTS Test] Audio converted to WAV, duration={len(audio_array)/sr:.2f}s")

                return JSONResponse(status_code=200, content={
                    "success": True,
                    "audio_base64": wav_base64,
                    "sample_rate": sr,
                    "duration": len(audio_array) / sr
                })

            except Exception as e:
                logger.error(f"[TTS Test] Error: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        # TTS 临时音频文件 API
        @fastapi.get('/tts/temp-files')
        async def list_tts_temp_files():
            """列出临时TTS音频文件"""
            try:
                temp_dir = os.path.join(DirectoryInfo.get_project_dir(), 'temp_tts_output')
                if not os.path.exists(temp_dir):
                    return JSONResponse(status_code=200, content={"files": []})

                files = []
                for ext in ['*.mp3', '*.wav', '*.pcm']:
                    files.extend(glob_module.glob(os.path.join(temp_dir, ext)))

                # 按修改时间排序，最新的在前
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

                file_list = []
                for f in files[:20]:  # 只返回最新的20个文件
                    file_list.append({
                        "name": os.path.basename(f),
                        "path": f,
                        "size": os.path.getsize(f),
                        "mtime": os.path.getmtime(f)
                    })

                return JSONResponse(status_code=200, content={"files": file_list})
            except Exception as e:
                logger.error(f"[TTS Temp Files] Error: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @fastapi.get('/tts/temp-files/{filename}')
        async def get_tts_temp_file(filename: str):
            """获取临时TTS音频文件"""
            try:
                temp_dir = os.path.join(DirectoryInfo.get_project_dir(), 'temp_tts_output')
                file_path = os.path.join(temp_dir, filename)

                # 安全检查：确保文件在临时目录内
                if not os.path.abspath(file_path).startswith(os.path.abspath(temp_dir)):
                    return JSONResponse(status_code=403, content={"error": "Access denied"})

                if not os.path.exists(file_path):
                    return JSONResponse(status_code=404, content={"error": "File not found"})

                # 根据扩展名确定 media_type
                ext = os.path.splitext(filename)[1].lower()
                media_types = {
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                    '.pcm': 'audio/pcm'
                }
                media_type = media_types.get(ext, 'application/octet-stream')

                return FileResponse(file_path, media_type=media_type, filename=filename)
            except Exception as e:
                logger.error(f"[TTS Temp File] Error: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @fastapi.get('/tts/temp-files-latest')
        async def get_latest_tts_temp_file():
            """获取最新的临时TTS音频文件"""
            try:
                temp_dir = os.path.join(DirectoryInfo.get_project_dir(), 'temp_tts_output')
                if not os.path.exists(temp_dir):
                    return JSONResponse(status_code=404, content={"error": "No temp files directory"})

                files = []
                for ext in ['*.mp3', '*.wav']:
                    files.extend(glob_module.glob(os.path.join(temp_dir, ext)))

                if not files:
                    return JSONResponse(status_code=404, content={"error": "No audio files found"})

                # 按修改时间排序，获取最新的
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_file = files[0]
                filename = os.path.basename(latest_file)

                ext = os.path.splitext(filename)[1].lower()
                media_types = {
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                }
                media_type = media_types.get(ext, 'audio/mpeg')

                return FileResponse(latest_file, media_type=media_type, filename=filename)
            except Exception as e:
                logger.error(f"[TTS Latest File] Error: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        frontend_path = Path(DirectoryInfo.get_src_dir() + '/handlers/client/rtc_client/frontend/dist')
        if frontend_path.exists():
            logger.info(f"Serving frontend from {frontend_path}")
            fastapi.mount('/ui', StaticFiles(directory=frontend_path), name="static")
            fastapi.add_route('/', RedirectResponse(url='/ui/index.html'))
        else:
            logger.warning(f"Frontend directory {frontend_path} does not exist")
            fastapi.add_route('/', RedirectResponse(url='/gradio'))

        if parent_block is None:
            parent_block = ui
        with ui:
            with parent_block:
                gradio.components.HTML(
                    """
                    <h1 id="openavatarchat">
                       The Gradio page is no longer available. Please use the openavatarchat-webui submodule instead.
                    </h1>
                    """,
                    visible=True
                )

    def on_setup_app(self, app: FastAPI, ui: gradio.blocks.Block, parent_block: Optional[gradio.blocks.Block] = None):
        avatar_config = {}
        self.setup_rtc_ui(ui, parent_block, app, avatar_config)

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        if not isinstance(handler_config, ClientRtcConfigModel):
            handler_config = ClientRtcConfigModel()
        context = ClientRtcContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def on_setup_session_delegate(self, session_context: SessionContext, handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
        handler_context = cast(ClientRtcContext, handler_context)
        session_delegate = cast(RtcClientSessionDelegate, session_delegate)

        session_delegate.timestamp_generator = session_context.get_timestamp
        session_delegate.data_submitter = handler_context.data_submitter
        session_delegate.input_data_definitions = self.output_bundle_definitions
        session_delegate.shared_states = session_context.shared_states

        handler_context.client_session_delegate = session_delegate

    def create_handler_detail(self, _session_context, _handler_context):
        inputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO
            ),
            ChatDataType.AVATAR_VIDEO: HandlerDataInfo(
                type=ChatDataType.AVATAR_VIDEO
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT
            ),
        }
        outputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
                definition=self.output_bundle_definitions[EngineChannelType.AUDIO]
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
                definition=self.output_bundle_definitions[EngineChannelType.VIDEO]
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=self.output_bundle_definitions[EngineChannelType.TEXT]
            ),
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs
        )

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        return self.create_handler_detail(session_context, context)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(ClientRtcContext, context)
        if context.client_session_delegate is None:
            return
        logger.debug(f"[RTC Client] Received data type: {inputs.type}, channel: {inputs.type.channel_type}")
        delegate = cast(RtcClientSessionDelegate, context.client_session_delegate)
        data_queue = delegate.output_queues.get(inputs.type.channel_type)
        if data_queue is not None:
            # Use thread-safe method to put data into the asyncio queue
            delegate._put_to_queue_threadsafe(data_queue, inputs)
            logger.debug(f"[RTC Client] Put data to queue (thread-safe)")

    def destroy_context(self, context: HandlerContext):
        pass
