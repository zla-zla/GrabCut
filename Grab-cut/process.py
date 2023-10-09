import os

dir = "./database/label/"
dir2 = "./database/"
filelist = os.listdir(dir)
total_num = len(filelist)
jpg = 1
png = 1
print('开始处理数据')
print(filelist)
# 遍历图片文件夹
for item in filelist:
    # 以jpg结尾的原始图片
    if item.endswith('.jpg'):
        src = dir + item
        if jpg<10:
            dst = dir2 + '0' + str(jpg) + '.jpg'
        else:
            dst = dir2 + str(jpg) + '.jpg'
        jpg+=1

        os.rename(src, dst)
        print
        'converting %s to %s ...' % (src, dst)

    # 以png结尾的掩码图像
    if item.endswith('.png'):
        src = dir + item
        if png<10:
            dst = dir2 + 'label_0' + str(png) + '.png'
        else:
            dst = dir2 + 'label_' + str(png) + '.png'
        dst = dir2 + 'label_' + str(png) + '.png'


        try:
            os.rename(src, dst)
            print
            'converting %s to %s ...' % (src, dst)

        except:
            print('失败')
            continue



