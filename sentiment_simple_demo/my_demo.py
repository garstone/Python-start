from classifier import SentimentClassifier
from flask import Flask, request, render_template
from time import time
from codecs import open
app = Flask(__name__)

print('Preparing the classifier...')
stime = time()
classifier = SentimentClassifier()
print('The classifier is ready')
print('That got', round(time() - stime, 2), 'seconds')


@app.route('/', methods=['GET', 'POST'])
def index_page(text='', prediction_message=''):
    if request.method == 'POST':
        text = request.form['text']
        prediction_message = classifier.get_prediction(text)

    return render_template('hello.html', text=text, prediction_message=prediction_message)


if __name__ == '__main__':
    app.run()
