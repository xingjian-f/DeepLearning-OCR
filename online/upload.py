from flask import Flask, request, render_template, make_response
from captcha import predict, beijing, guangdong, jiangsu, hubei, chi_single

app = Flask(__name__)
guangdong_model = guangdong()
print 'Model loaded 1.............................................................'
beijing_model = beijing()
print 'Model loaded 2.............................................................'
jiangsu_model = jiangsu()
print 'Model loaded 3.............................................................'
hubei_model = hubei()
print 'Model loaded 4.............................................................'
chi_single_model = chi_single()
print 'Model loaded 5.............................................................'

@app.route('/', methods=['GET', 'POST'])
def index():
	global guangdong_model, beijing_model, jiangsu_model, hubei_model, chi_single_model	
	if request.method == 'POST':
		imgs = request.files.to_dict()
		province = request.form['province']
		if province == 'guangdong':
			res = predict(guangdong_model, imgs)
		elif province == 'chinese':
			res = predict(chi_single_model, imgs)
		elif province == 'jiangsu':
			res = predict(jiangsu_model, imgs)
		elif province == 'nacao':
			res = 'Please use port:5002'
		elif province == 'beijing':
			res = predict(beijing_model, imgs)
		elif province == 'hubei':
			res = predict(hubei_model, imgs)
		else:
			res = 'No such province haha!'
		return res
	if request.method == 'GET':
		return render_template('index.html')

from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=5003)