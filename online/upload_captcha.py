import json
from flask import Flask, request, render_template, make_response
from captcha_new import predict
from models import beijing, guangdong, jiangsu, hubei, zhejiang, guizhou

app = Flask(__name__)
with open('pinyin') as f:
	pinyin = json.loads(f.readline())
print 'Etc files loaded ...........................................................'
guangdong_model = guangdong()
print 'Model loaded 1.............................................................'
beijing_model = beijing()
print 'Model loaded 2.............................................................'
jiangsu_model = jiangsu()
print 'Model loaded 3.............................................................'
hubei_model = hubei()
print 'Model loaded 4.............................................................'
zhejiang_model = zhejiang()
print 'Model loaded 5.............................................................'
guizhou_model = guizhou()

@app.route('/', methods=['GET', 'POST'])
def index():
	global guangdong_model, beijing_model, jiangsu_model, hubei_model
	global zhejiang_model, guizhou_model
	if request.method == 'POST':
		imgs = request.files.to_dict()
		province = request.form['province']
		if province == 'guangdong':
			res = predict(guangdong_model, imgs)
		elif province == 'guizhou':
			res = predict(guizhou_model, imgs)
		elif province == 'zhejiang':
			res = predict(zhejiang_model, imgs, pinyin)
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
	app.run(debug=0, host='0.0.0.0', port=5004)