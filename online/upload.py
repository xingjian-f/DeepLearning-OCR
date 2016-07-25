from flask import Flask, request, render_template, make_response
from captcha import predict, jiangsu, guangdong, beijing

app = Flask(__name__)
# guangdong_model = guangdong()
# jiangsu_model = jiangsu()
beijing_model = beijing()
@app.route('/', methods=['GET', 'POST'])
def index():
	global guangdong_model, jiangsu_model, beijing_model	
	if request.method == 'POST':
		imgs = request.files.to_dict()
		province = request.form['province']
		if province == 'guangdong':
			# res = predict(guangdong_model, imgs)
			res = 'Please use port:5000'
		elif province == 'jiangsu':
			# res = predict(jiangsu_model, imgs)
			res = 'Please use port:5000'
		elif province == 'beijing':
			res = predict(beijing_model, imgs)
		elif province == 'nacao':
			res = 'Please use port:5002'
		else:
			res = 'No such province haha!'
		return res
	if request.method == 'GET':
		return render_template('index.html')

from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=5001)