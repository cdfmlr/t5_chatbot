# To create a new Chatbot:
# - Chatbot: bussiness logic: holds and calls your model
#    - ask(prompt) -> response
# - ChatbotFactory: creates your Chatbot
#    - create_chatbot(config) ->
# - ChatbotConfig: config for ChatbotFactory & Chatbot

from dataclasses import dataclass
import json
import logging
from threading import Timer
import time
from typing import Dict
import uuid
from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import torch
from abc import ABCMeta, abstractmethod


class Chatbot(metaclass=ABCMeta):
    @abstractmethod
    def ask(self, session_id, prompt, **kwargs):
        """Ask Chatbot with prompt, return response text

        Raises:
            ChatbotError: Chatbot error
        """
        pass


# ChatbotConfig: {access_token, initial_prompt}
@dataclass
class ChatbotConfig:
    model: str = None
    initial_prompt: str = None

    @classmethod
    def from_json(cls, json_str: str):
        data: Dict = json.loads(json_str)
        c = cls(**data)
        return c


class ChatbotFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_chatbot(self, config: ChatbotConfig) -> Chatbot:
        """Chatbot factory"""
        raise NotImplementedError


class ChatbotProxy(Chatbot):
    """ChatbotProxy is a Chatbot (Factory + Proxy) used by MultiChatbot."""

    def __init__(self, session_id: str, config: ChatbotConfig, factory: ChatbotFactory, create_now=True):
        """A ChatbotProxy is represent to a session of MultiChatbot.
        (Maybe I should rename it ChatbotSession.)


        ChatbotProxy saves the config and use it to create a new 
        underlying ChatGTP instance.

        The ask() call will be proxy to the underlying Chatbot.

        renew() drops the underlying Chatbot and create a new one
        using the saved config. This is designed to kill a Chatbot
        session (to the openai's api), but keeps the session (w.r.t. 
        the MultiChatbot & the ChatbotServer).
        It avoids loooong conversations (which holding tons of history context)
        accumulates and costs tokens ($0.002 / 1K tokens) over and over again.
        """
        self.session_id = session_id
        self.config = config
        self.factory = factory

        self.initial_response = ""

        self.create_at = 0
        self.touch_at = 0

        self.Chatbot: Chatbot = None
        if create_now:
            self.renew()

    def renew(self):
        """re-create the underlying (real) Chatbot instance"""
        self.Chatbot = self.factory.create_chatbot(self.config)
        self.create_at = time.time()

    def is_timeout(self, timeout=900):
        """timeout: to be renew()"""
        return time.time() - self.create_at > timeout

    def is_zombie(self, timeout=1800):
        """zombie: do not renew()"""
        return time.time() - self.touch_at > timeout

    def ask(self, session_id, prompt, **kwargs):
        """ask the underlying (real) Chatbot"""
        self.touch_at = time.time()
        return self.Chatbot.ask(session_id, prompt, **kwargs)


# MultiChatbot: {session_id: Chatbot}:
#  - new(config) -> session_id
#  - ask(session_id, prompt) -> response
#  - delete(session_id)
class MultiChatbot(Chatbot):
    """MultiChatbot: {session_id: Chatbot}"""

    def __init__(self, chatbot_factory: ChatbotFactory, max_sessions=10, timeout=900, zombie_timeout=1800, check_timeout_interval=60):
        self.chatbot_factory = chatbot_factory
        self.chatbots: Dict[str, ChatbotProxy] = {}  # XXX: 话说这东西线程安全嘛

        self.max_sessions = max_sessions
        self.timeout = timeout  # timeout in seconds: 15 min
        self.zombie_timeout = zombie_timeout  # zombie timeout in seconds: 30 min
        # interval time to check timeout session in sec
        self.check_timeout_interval = check_timeout_interval

        Timer(self.check_timeout_interval, self.renew_timeout_sessions).start()

    def renew_timeout_sessions(self):
        now = time.time()
        for Chatbot in self.chatbots.values():
            if Chatbot.is_zombie(timeout=self.zombie_timeout):
                logging.debug(
                    f"MultiChatbot: zombie Chatbot: {Chatbot.session_id}, skip renew.")
                continue
            if Chatbot.is_timeout(timeout=self.timeout):
                logging.info(
                    f"MultiChatbot: renew a timeout Chatbot session {Chatbot.session_id}")
                Chatbot.renew()
        Timer(self.check_timeout_interval, self.renew_timeout_sessions).start()

    def clean_zombie_sessions(self):
        session_ids_to_del = []
        for Chatbot in self.chatbots.values():
            if Chatbot.is_zombie(timeout=self.timeout*2):
                session_ids_to_del.append(Chatbot.session_id)
        logging.info(
            f"MultiChatbot: delete zombie Chatbots: {session_ids_to_del}")
        for s in session_ids_to_del:
            self.delete(s)

    # raises TooManySessions, ChatbotError
    def new_session(self, config: ChatbotConfig) -> str:
        """Create new Chatbot session, return session_id

        session_id is an uuid4 string

        Raises:
            TooManySessions: Too many sessions
            ChatbotError: Chatbot error when asking initial prompt
        """
        if len(self.chatbots) >= self.max_sessions:
            self.clean_zombie_sessions()
        if len(self.chatbots) >= self.max_sessions:
            raise TooManySessions(self.max_sessions)

        session_id = str(uuid.uuid4())

        self.chatbots[session_id] = ChatbotProxy(
            session_id, config, self.chatbot_factory, create_now=True)

        return session_id

    def ask(self, session_id: str, prompt: str, **kwargs) -> str:  # raises ChatbotError
        """Ask Chatbot with session_id and prompt, return response text

        Raises:
            SessionNotFound: Session not found
            ChatbotError: Chatbot error when asking
        """
        if session_id not in self.chatbots:
            raise SessionNotFound(session_id)

        resp = self.chatbots[session_id].ask(session_id, prompt)

        return resp

    def delete(self, session_id: str):  # raises SessionNotFound
        """Delete Chatbot session

        Raises:
            SessionNotFound: Session not found
        """
        if session_id not in self.chatbots:
            raise SessionNotFound(session_id)

        del self.chatbots[session_id]


# Exceptions: TooManySessions, SessionNotFound, ChatbotError

class TooManySessions(Exception):
    def __init__(self, max_sessions: int):
        self.max_sessions = max_sessions
        self.message = f"Too many sessions, max {max_sessions}"
        super().__init__(self.message)


class SessionNotFound(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.message = f"Session {session_id} not found"
        super().__init__(self.message)


class ChatbotError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
