from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from flask_wtf.file import FileField
from flask_wtf.file import FileRequired
from wtforms import StringField
from wtforms.validators import DataRequired
from wtforms.validators import Regexp


class ImgForm(FlaskForm):
    fileimg = FileField(validators=[
        FileRequired(),
        FileAllowed(['png', 'jpg', 'jpeg', 'gif'])
    ])


class URLForm(FlaskForm):
    txturl = StringField(validators=[
        DataRequired(),
        Regexp(r'(?:http\:|https\:)?\/\/.*\.(?:png|jpg|jpeg|gif)$',
               message="不合法的图片url")])
