# 导入Flask类库
from flask import Flask, render_template, request, redirect
import json
from flask_cors import *
# pip install flask-cors i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
# pip install flask-cors i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
# pip install -i https://pypi.douban.com/simple --trusted-host pypi.douban.com fcntl
# https://pypi.tuna.tsinghua.edu.cn/simple
# gunicorn -c gunicorn.conf.py app:app
# 创建应用实例
app = Flask(__name__)
CORS(app, supports_credentials=True)
# 视图函数（路由）
@app.route('/user/<username>')
def say_hello(username):
	return '<h1>Hello %s !<h1>' % username

@app.route('/login', methods=['GET', 'POST'])
def login():
	return_dict = {'status': '0', 'key': 'none', 'msg': '这里是本地后端发回的消息'}
	# return_dict = {'status': '0', 'key': 'none', 'msg': '这里是阿里云服务器后端发回的消息'}
	get_data = request.args.to_dict()
	username = get_data.get("username")
	password = get_data.get("password")
	print(username, password)
	return json.dumps(return_dict, ensure_ascii=False)

# 启动实施（只在当前模块运行）
if __name__ == '__main__':
	# app.run(port=3333)
	app.run(port=1234)
