from flask import Flask, render_template, request, redirect, url_for
from predict import load_cnn, classify
import os

app = Flask(__name__)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['image']
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), image.filename)
        image.save(path)
        model, lb = load_cnn()
        name, prob = classify(path, model, lb)
        os.remove(path)
        return redirect(url_for('upload_file', name=name, prob=prob))
    else:
        name = request.args.get('name')
        prob = request.args.get('prob')
        return render_template('upload.html', name=name, prob=prob)



if __name__ == '__main__':
    app.run()