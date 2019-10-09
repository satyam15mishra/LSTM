import sample
from flask import Flask, render_template
import sys

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/text')
def gen_text():
	gen_text = sample.generated_text()
	return render_template('text_gen.html', gen_text = gen_text)

if __name__ == '__main__':
    app.run(debug = True)