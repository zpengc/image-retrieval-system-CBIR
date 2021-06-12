import os
from flask import current_app
from flask import flash
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from werkzeug.utils import secure_filename
from utils import download
from . import main
from .. import bof
from .forms import ImgForm
from .forms import URLForm
from utils import download_image_url


@main.route('/', methods=['GET', 'POST'])
def index():
    imgform = ImgForm()
    urlform = URLForm()

    # check if it is a POST request and if it is valid.
    if imgform.validate_on_submit():
        file = imgform.fileimg.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_DIR'], filename)
        if not os.path.exists(filepath):
            file.save(filepath)
        # In case blueprints are active you can shortcut references to the same
        # blueprint by prefixing the local endpoint with a dot(.).
        return redirect(url_for('.result', filename=filename))
    elif urlform.validate_on_submit():
        # image url, like https://icatcare.org/app/uploads/2018/07/Thinking-of-getting-a-cat.png
        url = urlform.txturl.data
        # filename, like Thinking-of-getting-a-cat.png
        filename = secure_filename(url.split('/')[-1])
        # E:\cbir_system\app/static/uploads\Thinking-of-getting-a-cat.png
        filepath = os.path.join(current_app.config['UPLOAD_DIR'], filename)
        # download(url, current_app.config['UPLOAD_DIR'], filename)
        download_image_url(url, filepath)
        if not os.path.exists(filepath):
            flash('无法下载指定URL的图片')
            return redirect(url_for('.index'))
        else:
            return redirect(url_for('.result', filename=filename))
    return render_template('index.html')


@main.route('/result', methods=['GET'])
def result():
    filename = request.args.get('filename')
    uri = os.path.join(current_app.config['UPLOAD_DIR'], filename)
    # images = bof.match(uri, top_k=20)
    images = bof.match(uri, top_k=10)
    return render_template('result.html', filename=filename, images=images)


# string: Accepts any text without a slash (the default).
# int: Accepts integers.
# float: Accepts numerical values containing decimal points.
# path: Similar to a string, but accepts slashes.

# as_attachment – set to True if you want to send this file with a Content-Disposition: attachment header.
# client-users can download file to the local host from server using this function

# show similar images as the result in the web
@main.route('/images/<path:file_dir>')
def expose_file(file_dir):
    print("显示图像：" + file_dir)
    return send_from_directory(current_app.config['BASE_DIR'], file_dir, as_attachment=True)
