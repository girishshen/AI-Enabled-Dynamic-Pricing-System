from flask import Flask


def register_blueprints(app: Flask) -> None:
    from .main_routes import main_bp

    app.register_blueprint(main_bp)