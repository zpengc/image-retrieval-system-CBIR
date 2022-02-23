import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from bof import BoF


bof = BoF()
csrf = CSRFProtect()


def create_app():
    app = Flask(__name__)
    # 配置csrf所需要的密钥
    app.secret_key = os.urandom(12)
    # 懒加载
    csrf.init_app(app)

    bof.init_app(app)

    # python relative import
    from .main import main

    # When you call .register_blueprint(), you apply all operations recorded in the Flask Blueprint main to app
    app.register_blueprint(main)
    return app
