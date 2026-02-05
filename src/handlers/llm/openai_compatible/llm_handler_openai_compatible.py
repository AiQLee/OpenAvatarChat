

import os
import re
import time
from typing import Dict, Optional, cast
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from openai import APIStatusError, OpenAI
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from handlers.llm.openai_compatible.chat_history_manager import ChatHistory, HistoryMessage


class LLMConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen-plus")
    system_prompt: str = Field(default="请你扮演一个 AI 助手，用简短的对话来回答用户的问题，并在对话内容中加入合适的标点符号，不需要加入标点符号相关的内容")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    api_url: str = Field(default=None)
    enable_video_input: bool = Field(default=False)
    history_length: int = Field(default=20)
    video_analysis_interval: float = Field(default=10.0)  # 每隔多少秒自动分析一次视频帧
    video_analysis_prompt: str = Field(default="请简短描述你在图片中看到了什么，用一两句话概括。")


class LLMContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.model_name = None
        self.system_prompt = None
        self.api_key = None
        self.api_url = None
        self.client = None
        self.input_texts = ""
        self.output_texts = ""
        self.current_image = None
        self.history = None
        self.enable_video_input = False
        self.video_analysis_interval = 10.0
        self.video_analysis_prompt = "请简短描述你在图片中看到了什么，用一两句话概括。"
        self.last_video_analysis_time = 0.0  # 上次分析视频的时间戳
        self.is_analyzing = False  # 防止重复分析
        self.shared_states = None  # 用于访问调试日志回调

    def send_debug_log(self, log_type: str, message: str, image_base64: str = None):
        """发送调试日志到前端"""
        if self.shared_states and self.shared_states.debug_log_callback:
            try:
                self.shared_states.debug_log_callback(log_type, message, image_base64)
            except Exception as e:
                logger.error(f"Failed to send debug log: {e}")


class HandlerLLM(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=LLMConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
            ),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, LLMConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, LLMConfig):
            handler_config = LLMConfig()
        context = LLMContext(session_context.session_info.session_id)
        context.model_name = handler_config.model_name
        context.system_prompt = {'role': 'system', 'content': handler_config.system_prompt}
        context.api_key = handler_config.api_key
        context.api_url = handler_config.api_url
        context.enable_video_input = handler_config.enable_video_input
        context.video_analysis_interval = handler_config.video_analysis_interval
        context.video_analysis_prompt = handler_config.video_analysis_prompt
        context.last_video_analysis_time = time.time()  # 初始化为当前时间，避免启动时立即分析
        context.shared_states = session_context.shared_states  # 保存 shared_states 引用用于调试日志
        context.history = ChatHistory(history_length=handler_config.history_length)
        context.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=context.api_key,
            base_url=context.api_url,
        )
        return context

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(LLMContext, context)
        text = None
        if inputs.type == ChatDataType.CAMERA_VIDEO and context.enable_video_input:
            image_data = inputs.data.get_main_data()
            logger.debug(f'[LLM] Received video frame, shape: {image_data.shape if image_data is not None else "None"}, enable_video_input: {context.enable_video_input}')
            context.current_image = image_data
            
            # 检查是否需要立即分析（用户切换到屏幕共享时）
            trigger_immediate = False
            if context.shared_states and context.shared_states.trigger_immediate_analysis:
                trigger_immediate = True
                context.shared_states.trigger_immediate_analysis = False
                logger.info('[Video Analysis] Immediate analysis triggered (video source changed)')

            # 检查是否需要定时分析视频帧
            current_time = time.time()
            time_since_last_analysis = current_time - context.last_video_analysis_time

            should_analyze = (
                trigger_immediate or
                (time_since_last_analysis >= context.video_analysis_interval)
            ) and not context.is_analyzing

            if should_analyze:
                context.is_analyzing = True
                context.last_video_analysis_time = current_time
                if trigger_immediate:
                    logger.info(f'[Video Analysis] Triggering immediate video analysis (source changed)')
                else:
                    logger.info(f'[Video Analysis] Triggering auto video analysis after {time_since_last_analysis:.1f}s')

                # 执行视频分析
                try:
                    yield from self._analyze_video_frame(context, output_definition)
                except Exception as e:
                    logger.error(f'[Video Analysis] Error: {e}', exc_info=True)
                finally:
                    context.is_analyzing = False
            else:
                logger.debug(f'[LLM] Video frame skipped - time_since_last: {time_since_last_analysis:.1f}s, interval: {context.video_analysis_interval}s, is_analyzing: {context.is_analyzing}')
            return
        elif inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if text is not None:
            context.input_texts += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        if len(chat_text) < 1:
            return
        logger.info(f'llm input {context.model_name} {chat_text} ')
        current_content = context.history.generate_next_messages(chat_text, 
                                                                 [context.current_image] if context.current_image is not None else [])
        logger.debug(f'llm input {context.model_name} {current_content} ')
        try:
            completion = context.client.chat.completions.create(
                model=context.model_name,  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[
                    context.system_prompt,
                ] + current_content,
                stream=True,
                stream_options={"include_usage": True}
            )
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            for chunk in completion:
                if (chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content):
                    output_text = chunk.choices[0].delta.content
                    context.output_texts += output_text
                    logger.info(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    yield output
            context.history.add_message(HistoryMessage(role="human", content=chat_text))
            context.history.add_message(HistoryMessage(role="avatar", content=context.output_texts))
        except Exception as e:
            logger.error(e)
            if (isinstance(e, APIStatusError)):
                response = e.body
                if isinstance(response, dict) and "message" in response:
                    response = f"{response['message']}"
            output_text = response 
            output = DataBundle(output_definition)
            output.set_main_data(output_text)
            output.add_meta("avatar_text_end", False)
            output.add_meta("speech_id", speech_id)
            yield output
        context.input_texts = ''
        context.output_texts = ''
        logger.info('avatar text end')
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def _analyze_video_frame(self, context: LLMContext, output_definition):
        """每隔一定时间自动分析视频帧并生成描述"""
        from engine_utils.media_utils import ImageUtils

        if context.current_image is None:
            logger.warning('[Video Analysis] No image available for analysis')
            context.send_debug_log('warning', '没有可用的图片进行分析')
            return

        logger.info(f'[Video Analysis] Analyzing current frame with prompt: {context.video_analysis_prompt}')
        logger.info(f'[Video Analysis] Image shape: {context.current_image.shape}, dtype: {context.current_image.dtype}')

        # 生成缩略图并发送到前端调试面板
        try:
            thumbnail = ImageUtils.numpy2thumbnail(context.current_image, max_size=150)
            context.send_debug_log(
                'video_analysis',
                f'📸 发送图片到 LLM 进行分析...\n提示词: {context.video_analysis_prompt}',
                thumbnail
            )
        except Exception as e:
            logger.error(f'Failed to generate thumbnail: {e}')
            context.send_debug_log('video_analysis', f'📸 发送图片到 LLM 进行分析...\n提示词: {context.video_analysis_prompt}')

        # 构建带图片的消息
        image_messages = context.history.generate_next_messages(
            context.video_analysis_prompt,
            [context.current_image]
        )

        try:
            completion = context.client.chat.completions.create(
                model=context.model_name,
                messages=[
                    context.system_prompt,
                ] + image_messages,
                stream=True,
                stream_options={"include_usage": True}
            )

            speech_id = f"video_analysis_{int(time.time())}"
            full_response = ""

            for chunk in completion:
                if chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content:
                    output_text = chunk.choices[0].delta.content
                    full_response += output_text
                    logger.info(f'[Video Analysis] {output_text}')
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    yield output

            # 添加到历史记录
            context.history.add_message(HistoryMessage(role="human", content=f"[视觉观察] {context.video_analysis_prompt}"))
            context.history.add_message(HistoryMessage(role="avatar", content=full_response))

            logger.info(f'[Video Analysis] Complete: {full_response}')
            context.send_debug_log('llm_response', f'🤖 LLM 回复: {full_response[:100]}{"..." if len(full_response) > 100 else ""}')

            # 发送结束标记
            end_output = DataBundle(output_definition)
            end_output.set_main_data('')
            end_output.add_meta("avatar_text_end", True)
            end_output.add_meta("speech_id", speech_id)
            yield end_output

        except Exception as e:
            logger.error(f'[Video Analysis] API Error: {e}')
            context.send_debug_log('error', f'❌ LLM API 错误: {str(e)}')
            raise

    def destroy_context(self, context: HandlerContext):
        pass

