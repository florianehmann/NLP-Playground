from flask import render_template
from app.base import bp


@bp.route('/')
@bp.route('/index')
def index():
    return render_template('base/index.html')
