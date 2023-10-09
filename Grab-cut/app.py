from flask import Flask, make_response, request,url_for, redirect, jsonify, g

from flask_cors import CORS  # 引用CORS，后期需要VUE支持跨域访问会用到
import pymysql
from sqlalchemy import and_
import os
from datetime import date, datetime
import json
# 读取图像，调用grabcut获取分割掩码，展示分割效果
import os
import cv2 as cv
import numpy as np
import time
from alg.grabCut2_fast import GrabCut as GMM_grab
from alg.grabCut_hist import GrabCut as Hist_grab


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BASE_DIR = os.path.join(BASE_DIR, 'Grab-cut', 'static')
IMG_ROOT = os.path.join(BASE_DIR, 'imgs')
RES_ROOT = os.path.join(BASE_DIR, 'tmp')


class JSONHelper():
    @staticmethod
    def jsonBQlist(bqlist):
        result=[]
        for item in bqlist:
            jsondata={}
            for i in range(item.__len__()):
                tdic={item._fields[i]:item[i]}
                jsondata.update(tdic)
            result.append(jsondata)
        return result

app = Flask(__name__)
CORS(app, resources=r'/*')
app.config['SECRET_KEY'] = '123456'



class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)


@app.route('/grabcut', methods=['GET', 'POST'])
def grabcut():
    top = request.form.get('top', type=int)  # formData格式区分
    left = request.form.get('left', type=int)  # formData格式区分
    width = request.form.get('width', type=int)  # formData格式区分
    height = request.form.get('height', type=int)  # formData格式区分a
    name = request.form.get('name', type=str)  # formData格式区分
    model = request.form.get('model', type=str)  # formData格式区分
    print(left, top, width, height, name, model)
    print(IMG_ROOT)
    print(os.path.join(IMG_ROOT, name))
    pred = str(int(time.time()))
    img = cv.imread(os.path.join(IMG_ROOT, name))
    rect = (left, top, width, height)  # 左下角坐标+宽和高
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    t = time.time()
    if model=="GMM":
        mask = GMM_grab(img, mask, rect)  # 调用grabcut算法
    else:
        mask = Hist_grab(img, mask, rect)  # 调用grabcut算法
    cost = time.time() - t
    print(f'本次分割总共用时为：{round(cost, 5)}s')

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    area = cv.bitwise_and(img, img, mask=mask2)  # mask=mask表示要提取的区域
    print(os.path.join(RES_ROOT, name))
    cv.imwrite('static/seg_result/'+pred+name, area)

    return_dict = {'time': str(round(time.time() - t, 3)), 'pred': pred, 'msg': 'OK'}

    return json.dumps(return_dict, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)










