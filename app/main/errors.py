from flask_wtf.csrf import CSRFError
from flask import render_template
from . import main

# errorhandler is a method inherited from Flask, not Blueprint. If you are using Blueprint, the equivalent is
# app_errorhandler.


@main.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html', reason=e.description), 400


@main.app_errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403


@main.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@main.app_errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
