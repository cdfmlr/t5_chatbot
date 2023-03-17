"""
Chinese T5 Pegasus Chatbot for muvtuber
"""

import argparse
import logging
from t5 import T5ChatbotFactory, T5ChatbotConfig
from muvtuber_chatbot_api import serve_grpc, MuvtuberGrpcServerConfig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--muvtb-grpc-serv", type=str, default="localhost:50053",
                        help="gRPC server address: host:port (e.g. localhost:50053)")
    args = parser.parse_args()

    config = MuvtuberGrpcServerConfig(
        chatbot_factory=T5ChatbotFactory(),
        chatbot_config_class=T5ChatbotConfig,
        max_sessions=10,
        address=args.muvtb_grpc_serv,
        timeout=60*60*24,
        zombie_timeout=60*60*25,
        check_timeout_interval=60*60,
        add_reflection_service=True)

    serve_grpc(config)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    main()
