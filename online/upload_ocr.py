from flask import Flask, request, render_template, make_response
from ocr import predict
from models import single_cha

app = Flask(__name__)
single_cha_model = single_cha()

@app.route('/', methods=['GET', 'POST'])
def index():
	global single_cha_model
	if request.method == 'POST':
		if 'province' in request.form:
			imgs = request.files.to_dict()
			types = 'file'
		else:
			imgs = request.form
			types = 'stringio'
		res = predict(single_cha_model, imgs, types)
		return res
	if request.method == 'GET':
		return render_template('index.html')


from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=5003)