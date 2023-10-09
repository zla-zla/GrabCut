<template>
  <div class="home">
    <div class="left">
        <el-upload  action="" :limit="20"   :on-change="handleChange"  :auto-upload="false"  :file-list="fileList" :show-file-list="false">
            <el-button class="btn" size="small" type="primary">上传图片</el-button>
        </el-upload>      
        <canvas id="canvas"></canvas>
        <div class="operate">
            <el-button class="btn" size="small" type="primary" @click="getCanvasObj">获取标注对象</el-button>
            <el-button class="btn" size="small" type="primary" @click="clear">清空</el-button>
            <el-button class="btn" size="small" type="primary" @click="submit">开始分割</el-button>
        </div>
        <div>
          <el-radio v-model="selectedOption" :label="option.value" v-for="option in options" :key="option.id">{{ option.label }}</el-radio>
          <!-- <p>你选择的选项是: {{ selectedOption }}</p> -->
        </div>        
    </div>

    <div class="right">
        <canvas id="result"></canvas>
        <div>
          本次分割累计用时为{{time}}s
        </div>
    </div>
  </div>
</template>


<script setup>
import { onMounted,ref } from "vue";
import { fabric } from "fabric";
import axios from 'axios';
import { ElRadio } from 'element-plus';
const selectedOption = ref('');
const options = [
  { id: 1, label: 'GMM模型', value: 'GMM' },
  { id: 2, label: '直方图模型', value: 'Hist' },
  // { id: 3, label: '选项3', value: 'option3' }
];

// 注册ElRadio组件
const components = {
  ElRadio
};

const time = ref('');
const pred = ref('');

// 监听键盘，按下a后开始绘制标注框
onMounted(() => {
  //监听键盘输入
  document.onkeydown = function () {
    let key = window.event.keyCode;
    if (key === 65) {
      //点击键盘 数字"1"后可以开始绘制标注框
      console.log(1111111);
      createRect();
      canvas.skipTargetFind = true;
    } else if (key === 66) {
        console.log(2222222)
        canvas.remove(canvas.getActiveObject())
    }
  };

  init();
});

/**
 * @name: 初始化
 * */
let canvas;
let res_canvas;

function init() {
  canvas = new fabric.Canvas("canvas", {
    backgroundColor: "rgb(100,100,200)", // 画布背景色
    selectionColor: "rgba(255,255,255,0.3)", // 画布中鼠标框选背景色
    selectionLineWidth: 0, // 画布中鼠标框选边框1
    // selection: false, // 在画布中鼠标是否可以框选 默认为true
  });
  res_canvas = new fabric.Canvas("result", {
    backgroundColor: "rgb(255,255,255)", // 画布背景色
    selectionColor: "rgba(255,255,255,0.3)", // 画布中鼠标框选背景色
    selectionLineWidth: 0, // 画布中鼠标框选边框1
    // selection: false, // 在画布中鼠标是否可以框选 默认为true
  });
  // createRect();
  insertImg();
  // showRes()
}

/**
 * @name: 绘制矩形
 */
let dtop = 0;
let dleft = 0;
let dw = 0;
let dh = 0;
let rect;
let imageName = '1.jpg'
let imgUrl = require("@/assets/resource/imgs/1.jpg"); //需要绘制的图片
let resUrl;

function createRect(row) {
  canvas.on("mouse:down", (options) => {
    dleft = options.e.offsetX;
    dtop = options.e.offsetY;
  });
  canvas.on("mouse:up", (options) => {
    let offsetX =
      options.e.offsetX > canvas.width ? canvas.width : options.e.offsetX;
    let offsetY =
      options.e.offsetY > canvas.height ? canvas.height : options.e.offsetY;

    dw = Math.abs(offsetX - dleft);
    dh = Math.abs(offsetY - dtop);
    // 拦截点击
    if (dw === 0 || dh === 0) {
      return;
    }

    rect = new fabric.Rect({
      top: dtop > offsetY ? offsetY : dtop,
      left: dleft > offsetX ? offsetX : dleft,
      width: dw,
      height: dh,
      fill: "rgba(101,169,230,0.2)",
      stroke: "rgb(101,169,230)", // 边框原色
      strokeWidth: 2, // 边框大小1
      // angle: 15,
      // selectable: false, // 是否允许当前对象被选中
      lockRotation: true, // 不允许旋转
    });
    console.log(rect);
    rect.set("strokeUniform", true); // 该属性在启用时可以防止笔划宽度受对象的比例值影响
    canvas.add(rect);
    stopDraw();

    canvas.skipTargetFind = false; //设置对象能否选中
  });

  canvas.on("mouse:move", (options) => {
    if (options.target) {
      objectMoving(options);
    }
  });
}

const insertImg = () => {
  // 插入背景
  let image = new Image();
  image.src = imgUrl;
  console.log(imgUrl);
  image.onload = () => {
    // 绘制图片
    // 设置canvas宽高
    canvas.setWidth(image.width);
    canvas.setHeight(image.height);

    fabric.Image.fromURL(imgUrl, (img) => {
      img.set({
        scaleX: 1,
        scaleY: 1,
        left: 0,
        top: 0,
      });
      canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
    });
  };
};

// 释放canvas监听
const stopDraw = () => {
  canvas.off("mouse:down");
  canvas.off("mouse:up");
};

// 限制对象的 不超出画布
function objectMoving(e) {
  var obj = e.target;
  if (!obj) return;
  if (
    obj.currentHeight > obj.canvas.height ||
    obj.currentWidth > obj.canvas.width
  ) {
    return;
  }
  obj.setCoords();
  // top-left corner
  if (obj.getBoundingRect().top < 0 || obj.getBoundingRect().left < 0) {
    obj.top = Math.max(obj.top, obj.top - obj.getBoundingRect().top);
    obj.left = Math.max(obj.left, obj.left - obj.getBoundingRect().left);
  }
  // bot-right corner
  if (
    obj.getBoundingRect().top + obj.getBoundingRect().height >
      obj.canvas.height ||
    obj.getBoundingRect().left + obj.getBoundingRect().width > obj.canvas.width
  ) {
    obj.top = Math.min(
      obj.top,
      obj.canvas.height -
        obj.getBoundingRect().height +
        obj.top -
        obj.getBoundingRect().top
    );
    obj.left = Math.min(
      obj.left,
      obj.canvas.width -
        obj.getBoundingRect().width +
        obj.left -
        obj.getBoundingRect().left
    );
    // obj.width = obj.getBoundingRect().width
    // obj.height = obj.getBoundingRect().height
  }
}

// 获取所有标注对象
function getCanvasObj() {
    console.log("canvas._objects", canvas._objects);

}

function clear(){
    canvas._objects.forEach((element) => {
        canvas.remove(element);
    })
}

function handleChange(file, fileLists){
			// console.log(file);
			// console.log(fileLists);
			// 本地服务器路径
      console.log('更换图片')
			let url=URL.createObjectURL(file.raw);
			// 本地电脑路径
			let url2=document.getElementsByClassName("el-upload__input")[0].value; 
      let pos = url2.lastIndexOf("\\") //'/所在的最后位置'
      imageName = url2.substr(pos + 1) //截取文件名称字符串
      imgUrl=url
      console.log(url)
      console.log(imageName)
      insertImg()
		}


function submit() {
    let rect = canvas._objects[0]
    const formData = new FormData();
    formData.append("top", parseInt(rect.top));
    formData.append("left", parseInt(rect.left));
    formData.append("width", rect.width);
    formData.append("height", rect.height);
    formData.append("name", imageName);
    formData.append("model",selectedOption.value)
    console.log(formData)
    // 发送 POST 请求到后端接口
    axios.post('http://127.0.0.1:5000/grabcut', formData)
        .then((response) => {
            console.log(response)
            time.value=response.data['time']
            pred.value = response.data['pred']
            console.log(time.value)
            showRes()
        })
        .catch((error) => {
            console.log(error)
        })
        
}

function showRes(){
    // console.log(imageName)
    // console.log('@/assets/resource/seg_result/'+imageName)
    // resUrl = require('@/assets/resource/seg_result/'+imageName); //需要绘制的图片
    resUrl='http://127.0.0.1:5000/static/seg_result/'+pred.value+imageName
    console.log(resUrl)
    let image = new Image();
    image.src = resUrl;
    console.log('结果图路径',resUrl);
    image.onload = () => {
        // 绘制图片
        // 设置canvas宽高
        res_canvas.setWidth(image.width);
        res_canvas.setHeight(image.height);

        fabric.Image.fromURL(resUrl, (img) => {
        img.set({
            scaleX: 1,
            scaleY: 1,
            left: 0,
            top: 0,
        });
        res_canvas.setBackgroundImage(img, res_canvas.renderAll.bind(res_canvas));
        });
    };
}

</script>

<style scoped>
.home {
  width: 100%;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}
#canvas {
    display: block;
    margin: 0 auto;

}
.left{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  width:50%
}
.right{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  width:50%
}
.operate {
  margin-top: 30px;
}
.btn {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  line-height: 1;
  height: 32px;
  width: 150px;
  white-space: nowrap;
  cursor: pointer;
  padding: 8px 15px;
  border: 1px solid #ddd;
}
</style>



