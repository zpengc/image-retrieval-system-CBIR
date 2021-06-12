from flask import Blueprint

# create a blueprint
# The first argument is the Blueprint’s name, which is used by Flask’s routing mechanism.
# The second argument is the Blueprint’s import name, which Flask uses to locate the Blueprint’s resources.
main = Blueprint('main', __name__)
# 避免循环导入依赖
from . import views, errors
