"""
豆包(火山引擎)TTS Handler - V3 API版本
基于火山引擎OpenSpeech V3 API实现的语音合成handler

API文档: https://www.volcengine.com/docs/6561/1096680
"""
import io
import json
import uuid
import base64
import re
import os
import time
from datetime import datetime
from typing import Dict, Optional, List, cast

import requests
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class DoubaoTTSConfig(HandlerBaseConfigModel, BaseModel):
    """豆包TTS配置 - V3 API"""
    appid: str = Field(default=os.getenv("DOUBAO_APPID", ""), description="火山引擎应用ID")
    access_token: str = Field(default=os.getenv("DOUBAO_ACCESS_TOKEN", ""), description="访问令牌")
    resource_id: str = Field(default="volc.service_type.10029", description="资源ID")
    voice_type: str = Field(default="zh_female_cancan_mars_bigtts", description="音色类型")
    speed_ratio: float = Field(default=1.0, description="语速比例 0.5-2.0")
    volume_ratio: float = Field(default=1.0, description="音量比例 0.5-2.0")
    pitch_ratio: float = Field(default=1.0, description="音调比例 0.5-2.0")
    encoding: str = Field(default="mp3", description="音频编码格式: pcm, mp3, wav")
    sample_rate: int = Field(default=24000, description="采样率")


class DoubaoTTSContext(HandlerContext):
    """豆包TTS上下文"""
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[DoubaoTTSConfig] = None
        self.input_text: str = ''


class HandlerDoubaoTTS(HandlerBase, ABC):
    """豆包TTS Handler - V3 API版本"""

    # 火山引擎TTS V3 API端点 (HTTP单向流式)
    API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

    # 可用的音色列表 (豆包语音合成模型2.0)
    VOICE_TYPES: List[str] = [
        "zh_female_vv_uranus_bigtts",              # Vivi 2.0 - 通用场景
        "zh_female_xiaohe_uranus_bigtts",          # 小何 2.0 - 通用场景
        "zh_male_m191_uranus_bigtts",              # 云舟 2.0 - 通用场景
        "zh_male_taocheng_uranus_bigtts",          # 小天 2.0 - 通用场景
        "zh_female_xueayi_saturn_bigtts",          # 儿童绘本 - 有声阅读
        "zh_male_dayi_saturn_bigtts",              # 大壹 - 视频配音
        "zh_female_mizai_saturn_bigtts",           # 黑猫侦探社咪 - 视频配音
        "zh_female_jitangnv_saturn_bigtts",        # 鸡汤女 - 视频配音
        "zh_female_meilinvyou_saturn_bigtts",      # 魅力女友 - 视频配音
        "zh_female_santongyongns_saturn_bigtts",   # 流畅女声 - 视频配音
        "zh_male_ruyayichen_saturn_bigtts",        # 儒雅逸辰 - 视频配音
        "saturn_zh_female_keainvsheng_tob",        # 可爱女生 - 角色扮演
        "saturn_zh_female_tiaopigongzhu_tob",      # 调皮公主 - 角色扮演
        "saturn_zh_male_shuanglangshaonian_tob",   # 爽朗少年 - 角色扮演
        "saturn_zh_male_tiancaitongzhuo_tob",      # 天才同桌 - 角色扮演
        "saturn_zh_female_cancan_tob",             # 知性灿灿 - 角色扮演
    ]

    # 音色切换间隔(秒)
    VOICE_SWITCH_INTERVAL = 10.0

    def __init__(self):
        super().__init__()
        self.config: Optional[DoubaoTTSConfig] = None
        self._current_voice_index: int = 0
        self._last_voice_switch_time: float = 0.0

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=DoubaoTTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 1, self.config.sample_rate
        ))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        self.config = cast(DoubaoTTSConfig, handler_config)
        logger.info(f"[Doubao TTS] Loaded with voice_type: {self.config.voice_type}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        if not isinstance(handler_config, DoubaoTTSConfig):
            handler_config = DoubaoTTSConfig()
        context = DoubaoTTSContext(session_context.session_info.session_id)
        context.config = handler_config
        context.input_text = ''
        return context

    def start_context(self, session_context: SessionContext, context: HandlerContext):
        pass

    def filter_text(self, text: str) -> str:
        """过滤文本中的特殊字符"""
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"
        filtered_text = re.sub(pattern, "", text)
        return filtered_text

    def _get_current_voice_type(self) -> str:
        """获取当前音色

        注意: 音色轮换功能暂时禁用，因为不同音色需要不同的resource_id。
        当前resource_id (volc.service_type.10029) 只支持 *_mars_bigtts 类型的音色。
        如需启用音色轮换，需要确保所有音色与resource_id匹配。
        """
        # 暂时禁用音色轮换，使用配置的voice_type
        current_voice = self.config.voice_type
        logger.debug(f"[Doubao TTS] Using configured voice_type: {current_voice}")
        return current_voice

        # === 以下是音色轮换代码，暂时禁用 ===
        # current_time = time.time()
        # # 检查是否需要切换音色
        # if current_time - self._last_voice_switch_time >= self.VOICE_SWITCH_INTERVAL:
        #     self._last_voice_switch_time = current_time
        #     self._current_voice_index = (self._current_voice_index + 1) % len(self.VOICE_TYPES)
        #     current_voice = self.VOICE_TYPES[self._current_voice_index]
        #     logger.info(f"[Doubao TTS] Voice switched to: {current_voice} (index: {self._current_voice_index})")
        # current_voice = self.VOICE_TYPES[self._current_voice_index]
        # logger.debug(f"[Doubao TTS] Current voice_type: {current_voice}")
        # return current_voice

    def _call_tts_api(self, text: str) -> Optional[np.ndarray]:
        """调用豆包TTS V3 API"""
        try:
            # 获取当前音色(每10秒自动切换)
            current_voice = self._get_current_voice_type()

            # V3 API Headers格式
            headers = {
                "X-Api-App-Id": self.config.appid,
                "X-Api-Access-Key": self.config.access_token,
                "X-Api-Resource-Id": self.config.resource_id,
                "Content-Type": "application/json",
                "Connection": "keep-alive"
            }

            # V3 API 请求体格式
            request_body = {
                "user": {
                    "uid": "openavatar_user"
                },
                "req_params": {
                    "text": text,
                    "speaker": current_voice,
                    "audio_params": {
                        "format": self.config.encoding,
                        "sample_rate": self.config.sample_rate,
                        "enable_timestamp": False
                    },
                    "additions": json.dumps({
                        "explicit_language": "zh",
                        "disable_markdown_filter": True
                    })
                }
            }

            logger.debug(f"[Doubao TTS] Calling V3 API for text: {text[:50]}...")

            # 使用流式响应
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=request_body,
                timeout=30,
                stream=True
            )

            if response.status_code != 200:
                logger.error(f"[Doubao TTS] API error: {response.status_code} - {response.text[:500]}")
                return None

            # 解析流式响应，收集所有音频数据
            audio_data = bytearray()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    code = data.get("code", 0)

                    # 正常音频数据
                    if code == 0 and "data" in data and data["data"]:
                        chunk_audio = base64.b64decode(data["data"])
                        audio_data.extend(chunk_audio)
                        continue

                    # 结束标记
                    if code == 20000000:
                        logger.debug(f"[Doubao TTS] TTS completed")
                        break

                    # 错误
                    if code > 0 and code != 20000000:
                        logger.error(f"[Doubao TTS] API error in stream: {data}")
                        return None
                except json.JSONDecodeError as e:
                    logger.warning(f"[Doubao TTS] Failed to parse line: {line[:100]}, error: {e}")

            if not audio_data:
                logger.error("[Doubao TTS] No audio data received")
                return None

            audio_bytes = bytes(audio_data)
            logger.debug(f"[Doubao TTS] Received {len(audio_bytes)} bytes of audio")

            # 保存音频到临时文件夹用于调试
            temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "temp_tts_output")
            temp_dir = os.path.abspath(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ext = self.config.encoding if self.config.encoding in ["mp3", "wav"] else "pcm"
            temp_file = os.path.join(temp_dir, f"tts_{timestamp}.{ext}")
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"[Doubao TTS] Audio saved to: {temp_file}")

            # 将音频数据转换为numpy数组
            if self.config.encoding == "pcm":
                # PCM 16bit signed integer
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                # 归一化到 [-1, 1]
                audio_array = audio_array / 32768.0
            else:
                # 对于mp3/wav格式，需要用librosa解码
                import librosa
                audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.config.sample_rate)

            logger.info(f"[Doubao TTS] Generated audio shape: {audio_array.shape}")
            return audio_array

        except Exception as e:
            logger.error(f"[Doubao TTS] Error calling API: {e}")
            import traceback
            traceback.print_exc()
            return None

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        logger.info(f"[Doubao TTS] handle() called, input type: {inputs.type}")
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(DoubaoTTSContext, context)

        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
            logger.info(f"[Doubao TTS] Received AVATAR_TEXT: {text[:100] if text else 'None'}...")
        else:
            logger.debug(f"[Doubao TTS] Ignoring input type: {inputs.type}")
            return

        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False)

        if not text_end:
            # 按标点符号分割句子
            sentences = re.split(r'(?<=[,.~!?，。！？])', context.input_text)
            if len(sentences) > 1:
                complete_sentences = sentences[:-1]
                context.input_text = sentences[-1]

                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue

                    logger.info(f'[Doubao TTS] Processing sentence: {sentence}')

                    audio_array = self._call_tts_api(sentence)
                    if audio_array is not None:
                        audio_array = audio_array[np.newaxis, ...]
                        output = DataBundle(output_definition)
                        output.set_main_data(audio_array)
                        output.add_meta("avatar_speech_end", False)
                        output.add_meta("speech_id", speech_id)
                        logger.info(f'[Doubao TTS] Submitting audio data for sentence')
                        context.submit_data(output)
        else:
            # 处理最后一段文本
            logger.info(f'[Doubao TTS] Last sentence: {context.input_text}')
            if context.input_text and len(context.input_text.strip()) > 0:
                audio_array = self._call_tts_api(context.input_text)
                if audio_array is not None:
                    audio_array = audio_array[np.newaxis, ...]
                    output = DataBundle(output_definition)
                    output.set_main_data(audio_array)
                    output.add_meta("avatar_speech_end", False)
                    output.add_meta("speech_id", speech_id)
                    context.submit_data(output)

            context.input_text = ''

            # 发送结束信号
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            output.add_meta("avatar_speech_end", True)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            logger.info(f"[Doubao TTS] Speech end")

    def destroy_context(self, context: HandlerContext):
        logger.info('[Doubao TTS] Destroy context')
