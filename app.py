import logging
import os
from logging.handlers import RotatingFileHandler
from flask import Flask

from routes import register_blueprints


def create_app() -> Flask:
    base_dir = os.path.abspath(os.path.dirname(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )

    # Ensure directories exist
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data_entries"), exist_ok=True)

    # Configure logging
    log_path = os.path.join(base_dir, "logs", "app.log")
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)
    # Avoid duplicate handlers when reloaded
    if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
        app.logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in app.logger.handlers):
        app.logger.addHandler(stream_handler)

    register_blueprints(app)

    app.logger.info("DynamicPricingSystem Flask app initialized")

    return app


app = create_app()


if __name__ == "__main__":
    # Run locally: python app.py
    print("\nDynamicPricingSystem running at http://127.0.0.1:5000\n")
    app.run(debug=True, host="127.0.0.1", port=5000)