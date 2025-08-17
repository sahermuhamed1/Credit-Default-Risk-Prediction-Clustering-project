from flask import Flask
import os

def create_app():
    # Explicitly set template folder to prevent TemplateNotFound issues
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    app = Flask(__name__, template_folder=template_dir)

    # Import and register blueprints or routes here
    from app import routes
    app.register_blueprint(routes.bp)

    return app