from flask import flash, render_template
from app.base import bp
from app.base.forms import QueryForm


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
def index():
    form = QueryForm()
    result = None

    if form.validate_on_submit():
        result = form.tweet_text.data

    return render_template('base/index.html', form=form, result=result)
