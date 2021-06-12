from flask import Flask
from flask_wtf.csrf import CSRFProtect
from config import config
from bof import BoF


bof = BoF()
csrf = CSRFProtect()


def create_app(config_name):
    # creates the Flask instance
    app = Flask(__name__)
    print("creates the Flask instance", app)

    # loads the configuration from the config[config_name] module
    app.config.from_object(config[config_name])
    # app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

    # use flask extensions
    # init_app function is used to control the integration of a package to one or more Flask applications
    config[config_name].init_app(app)
    bof.init_app(app)
    csrf.init_app(app)

    # python relative import
    from .main import main

    # register a blueprint
    # When you call .register_blueprint(), you apply all operations recorded in the Flask Blueprint main to app
    app.register_blueprint(main)
    return app
