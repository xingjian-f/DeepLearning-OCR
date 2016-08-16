from flask import Flask, request, render_template, make_response
from ocr import predict
from models import chi_single

app = Flask(__name__)
chi_single_model = chi_single()

@app.route('/', methods=['GET', 'POST'])
def index():
	global chi_single_model
	if request.method == 'POST':
		if 'province' in request.form:
			imgs = request.files.to_dict()
			types = 'file'
		else:
			imgs = request.form
			types = 'stringio'
		res = predict(chi_single_model, imgs, types)
		return res
	if request.method == 'GET':
		return render_template('index.html')


from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=5003)