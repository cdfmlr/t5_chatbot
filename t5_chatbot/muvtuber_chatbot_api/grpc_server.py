from dataclasses import dataclass
import logging
from concurrent import futures
from typing import Type
import grpc

from .protos import chatbot_pb2, chatbot_pb2_grpc
from .cooldown import CooldownException
from .chatbot import MultiChatbot, ChatbotFactory, ChatbotConfig, ChatbotError, TooManySessions, SessionNotFound


class ChatbotGrpcServer(chatbot_pb2_grpc.ChatbotServiceServicer):
    def __init__(self, multichatbot: MultiChatbot, chatbot_config: ChatbotConfig):
        self.multichatbot = multichatbot
        self.chatbot_config = chatbot_config

    def NewSession(self, request, context):
        """NewSession creates a new session with Chatbot.
        Input: access_token (string) and initial_prompt (string).
        Output: session_id (string).
        """
        # read config
        try:
            config = self.chatbot_config.from_json(request.config)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            logging.warning(
                f'ChatbotGrpcServer.NewSession: bad config: {str(e)}')
            return chatbot_pb2.NewSessionResponse()

        # new session
        session_id = None
        try:
            session_id = self.multichatbot.new_session(config)
        except TooManySessions as e:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(str(e))
        except ChatbotError as e:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))

        if context.code() != grpc.StatusCode.OK and context.code() != None:
            logging.warn(
                f'ChatbotGrpcServer.NewSession: {context.code()}: {context.details()}')
        else:
            logging.info(
                f'ChatbotGrpcServer.NewSession: (OK) session_id={session_id}')

        # XXX: deprecate initial_response?
        initial_response = None
        try:
            initial_response = self.multichatbot.chatgpts[session_id].initial_response
        except:
            pass
        return chatbot_pb2.NewSessionResponse(session_id=session_id, initial_response=initial_response)

    def Chat(self, request, context):
        """Chat sends a prompt to Chatbot and receives a response.
        Input: session_id (string) and prompt (string).
        Output: response (string).
        """
        if not request.session_id:
            # raise ValueError('session_id is required')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('session_id is required')
            logging.warn('ChatbotGrpcServer.Chat: session_id is required')
            return chatbot_pb2.ChatResponse()
        if not request.prompt:
            # raise ValueError('prompt is required')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('prompt is required')
            logging.warn('ChatbotGrpcServer.Chat: prompt is required')
            return chatbot_pb2.ChatResponse()

        response = None
        try:
            response = self.multichatbot.ask(
                request.session_id, request.prompt)
        except SessionNotFound as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
        except ChatbotError as e:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
        except CooldownException as e:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(str(e))

        if context.code() != grpc.StatusCode.OK and context.code() != None:
            logging.warn(
                f'ChatbotGrpcServer.Chat: ({context.code()}) {context.details()}')
        else:
            logging.info(
                f'ChatbotGrpcServer.Chat: (OK) {response}')

        return chatbot_pb2.ChatResponse(response=response)

    def DeleteSession(self, request, context):
        """DeleteSession deletes a session with Chatbot.
        Input: session_id (string).
        Output: session_id (string).
        """
        if not request.session_id:
            # raise ValueError('session_id is required')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('session_id is required')
            logging.warn(
                'ChatbotGrpcServer.DeleteSession: session_id is required')
            return chatbot_pb2.DeleteSessionResponse()

        try:
            self.multichatbot.delete(request.session_id)
        except SessionNotFound as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))

        if context.code() != grpc.StatusCode.OK and context.code() != None:
            logging.warn(
                f'ChatbotGrpcServer.DeleteSession: ({context.code()}) {context.details()}')
        else:
            logging.info(
                f'ChatbotGrpcServer.DeleteSession: (OK) {request.session_id}')

        return chatbot_pb2.DeleteSessionResponse(session_id=request.session_id)


@dataclass
class MuvtuberGrpcServerConfig():
    chatbot_factory: ChatbotFactory
    chatbot_config_class: Type[ChatbotConfig]
    max_sessions: int = 10
    address: str = 'localhost:50052'
    timeout: int = 60*60  # seconds
    zombie_timeout: int = 60*60*2  # seconds
    check_timeout_interval: int = 60  # seconds
    add_reflection_service: bool = True


def serve_grpc(config: MuvtuberGrpcServerConfig):
    """Starts a gRPC server at the specified address 'host:port'."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    multichatbot = MultiChatbot(config.chatbot_factory,
                                max_sessions=config.max_sessions,
                                timeout=config.timeout,
                                zombie_timeout=config.zombie_timeout,
                                check_timeout_interval=config.check_timeout_interval)

    chatbot_grpc_server = ChatbotGrpcServer(
        multichatbot, config.chatbot_config_class())

    chatbot_pb2_grpc.add_ChatbotServiceServicer_to_server(
        chatbot_grpc_server, server)

    SERVICE_NAMES = [
        chatbot_pb2.DESCRIPTOR.services_by_name['ChatbotService'].full_name]

    if config.add_reflection_service:
        from grpc_reflection.v1alpha import reflection
        SERVICE_NAMES.append(reflection.SERVICE_NAME)
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        logging.info(f'gRPC reflection enabled.')

    server.add_insecure_port(config.address)
    server.start()
    print(f'Chatbot gRPC server started at {config.address}.')
    print(f'Services: {SERVICE_NAMES}')
    server.wait_for_termination()
