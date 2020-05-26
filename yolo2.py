import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(2234)
np.random.seed(2234)

print(tf.__version__)

print(tf.test.is_gpu_available())

import xml.etree.ElementTree as ET
def parse_annotation(img_dir, ann_dir, labels):
    # img_dir: image path
    # ann_dir: annotation xml file path
    # labels: ('sugarweet', 'weed')
    # parse annotation info from xml file
    """
    <annotation>
        <filename>X2-10-0.png</filename>
        <size>
            <width>512</width>
            <height>512</height>
            <depth>3</depth>
        </size>
        <object>
            <name>sugarbeet</name>
            <bndbox>
                <xmin>1</xmin>
                <ymin>230</ymin>
                <xmax>39</xmax>
                <ymax>300</ymax>
            </bndbox>
        </object>
    """
    imgs_info = []

    max_boxes = 0

    # for each annotation xml file
    for ann in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann))

        img_info = dict()

        img_info['object'] = []

        boxes_counter = 0
        for elem in tree.iter():
            
            if 'filename' in elem.tag:
                img_info['filename'] = os.path.join(img_dir,elem.text)

            if 'width' in elem.tag:
                img_info['width'] = int(elem.text)
                assert img_info['width'] == 512

            if 'height' in elem.tag:
                img_info['height'] = int(elem.text)
                assert img_info['height'] == 512

            if 'object' in elem.tag or 'part' in elem.tag:
                # x1-y1-x2-y2-label
                boxes_counter += 1
                object_info = [0,0,0,0,0]
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = labels.index(attr.text) + 1
                        # 0背景,1,2物体
                        object_info[4] = label
                    if 'bndbox' in attr.tag:
                        for pos in list(attr):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)

        # 处理完1张图片后

        imgs_info.append(img_info) # filename, w/h/box_info

        # (N, 5) = (max_objects_num, 5)
        if boxes_counter > max_boxes:
            max_boxes = boxes_counter

    # 处理完所有图片后

    # the maximum boxes number is max_boxes
    boxes = np.zeros([len(imgs_info), max_boxes, 5])
    imgs_path = [] # filename list

    for i, img_info in enumerate(imgs_info):
        # [N, 5] img_info['object']
        img_boxes = np.array(img_info['object'])
        boxes[i, :img_boxes.shape[0]] = img_boxes
        imgs_path.append(img_info['filename'])

    # imgs_path: list of image path
    # boxes: [b, 40, 5]
    return imgs_path, boxes 

obj_names = ('sugarbeet', 'weed')
imgs_path, boxes = parse_annotation('data/train/image', 'data/train/annotation', obj_names)

# 1.2 get dataset

def preprocess(img, img_boxes):
    # img: string
    # img_boxes: [40, 5]
    
    x = tf.io.read_file(img)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32) # 0-255 -> 0.0-1.0

    return x, img_boxes

def get_dataset(img_dir, ann_dir, batchsz):
    # return tf dataset
    # [b], boxes [b, 40, 5]
    imgs_path, boxes = parse_annotation(img_dir, ann_dir, obj_names)
    db = tf.data.Dataset.from_tensor_slices((imgs_path, boxes))
    db = db.shuffle(1000).map(preprocess).batch(batchsz).repeat()

    print('db Images:', len(imgs_path))
    
    return db

train_db = get_dataset('data/train/image', 'data/train/annotation', 4)
print(train_db)

# 1.3 visual the db
import matplotlib.pyplot as plt
from matplotlib import patches

def db_visualize(db):
    # imgs: [b, 512, 512, 3]
    # imgs_boxes: [b, 40, 5]
    imgs, imgs_boxes = next(iter(db))
    img, img_boxes = imgs[0], imgs_boxes[0]

    f, ax1 = plt.subplots(1, figsize=(10, 10))
    # display the image, [512, 512, 3]
    ax1.imshow(img)
    
    for x1, y1, x2, y2, l in img_boxes: # [40, 5]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1
        h = y2 - y1

        # (R, G, B)
        if l==1: # green for sugarbeet
            color = (0, 1, 0)
        elif l==2: # red for weed
            color = (1, 0, 0)
        else: # ignore invalid boxes
            continue

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')

        ax1.add_patch(rect)

db_visualize(train_db)

# 1.4 data augmentation 数据增强
# imgaug 此库专做数据增强
import imgaug as ia
from    imgaug import augmenters as iaa
def augmentation_generator(yolo_dataset):
    '''
    Augmented batch generator from a yolo dataset

    Parameters
    ----------
    - YOLO dataset
    
    Returns
    -------
    - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    '''
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                       y1=bb[1],
                                       x2=bb[2],
                                       y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] +bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(512, 512)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)), # change brightness
            ])
        #seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i,j,0] = bb.x1
                boxes[i,j,1] = bb.y1
                boxes[i,j,2] = bb.x2
                boxes[i,j,3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        #batch = (img_aug, boxes)
        yield batch

# test
aug_train_db = augmentation_generator(train_db)
db_visualize(aug_train_db)


# 2.1 GT box
IMGSZ = 512 # 512 * 512 
GRIDSZ = 16 # 16 * 16

# [w, h, w, h, w, h...]
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
# ANCHORS_NUM = len(ANCHORS) // 2

def process_true_boxes(gt_boxes, anchors):
    # gt_boxes: [40, 5]
    # 512 // 16 = 32
    scale = IMGSZ // GRIDSZ
    # [5, 2]
    anchors = np.array(anchors).reshape((5, 2))

    detector_mask = np.zeros([GRIDSZ, GRIDSZ, 5, 1])
    # 5：x，y, w, h, l(label:0, 1, 2)
    matching_gt_box = np.zeros([GRIDSZ, GRIDSZ, 5, 5])
    # [40, 5] x1,y1,x2,y2,l -> x,y,w,h,l 左上角点与右下角点 => 中心点和宽高
    gt_boxes_grid = np.zeros(gt_boxes.shape)

    # DB: tensor => numpy
    gt_boxes = gt_boxes.numpy()

    for i, box in enumerate(gt_boxes): # [40, 5]
        # box: [5],  x1,y1,x2,y2,l
        # 512 => 16
        x = ((box[0] + box[2]) / 2) / scale
        y = ((box[1] + box[3]) / 2) / scale
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale
        # [40, 5] x,y,w,h,l 
        gt_boxes_grid[i] = np.array([x, y, w, h, box[4]])

        if w * h > 0: # valid box
            # x, y: 7.3, 6.8
            best_anchor = 0
            best_iou = 0
            for j in range(5):
                intersection = np.minimum(w, anchors[j, 0]) * np.minimum(h, anchors[j, 1])
                union = w * h + (anchors[j, 0] * anchors[j, 1]) - intersection
                iou = intersection / union

                if iou > best_iou: # best iou
                    best_anchor = j
                    best_iou = iou

            # found the best anchors
            # 两矩形，相交大于0，不相交等于0
            if best_iou > 0:
                # np.floor() 向下取整 1.1 => 1. 下，小于，坐标轴左边，-2.2 => -3.
                x_coord = np.floor(x).astype(np.int32)
                y_coord = np.floor(y).astype(np.int32)
                # [h, w, 5, l]
                detector_mask[y_coord, x_coord, best_anchor] = 1
                # [h, w, 5, x-y-w-h-l]
                matching_gt_box[y_coord, x_coord, best_anchor] = \
                    np.array([x, y, w, h, box[4]])

    # [40, 5] => [16, 16, 5, 5]
    # [16, 16, 5, 5] 真实标签，只有iou最高的box上有数字
    # [16, 16, 5, 1] 可从上面推测，不过还是单独提取出来
    # [40, 5] x1,y1,x2,y2,l -> x,y,w,h,l 后，且经过缩放scale后的gt_box
    return matching_gt_box, detector_mask, gt_boxes_grid

# 2.2 
def ground_truth_generator(db):

    for imgs, imgs_boxes in db:
        # db: 可迭代类型
        # imgs: [b, 512, 512, 3] 经过数据增强后的图片
        # imgs_boxes: [b, 40, 5] 原始的boxes
        batch_matching_gt_boxes = []
        batch_detector_mask = []
        batch_gt_boxes_grid = []

        b = imgs.shape[0]
        for i in range(b): # for each image
            matching_gt_box, detector_mask, gt_boxes_grid = \
                process_true_boxes(imgs_boxes[i], ANCHORS)

            batch_matching_gt_boxes.append(matching_gt_box)
            batch_detector_mask.append(detector_mask)
            batch_gt_boxes_grid.append(gt_boxes_grid)

        # matching_gt_box, detector_mask, gt_boxes_grid，重用它们的名字

        # [b, 16, 16, 5, 1]
        detector_mask = tf.cast(np.array(batch_detector_mask), dtype=tf.float32)
        # [b, 16, 16, 5, 5] x-y-w-h-l
        matching_gt_box = tf.cast(np.array(batch_matching_gt_boxes), dtype=tf.float32)
        # [b, 40, 5] x-y-w-h-l
        gt_boxes_grid = tf.cast(np.array(batch_gt_boxes_grid), dtype=tf.float32)

        # [b, 16, 16, 5]
        # 4为第5个，即label，全部读取的，有0，1，2
        matching_classes = tf.cast(matching_gt_box[..., 4], dtype=tf.int32)
        # [b, 16, 16, 5, 3]
        matching_classes_one_hot = tf.one_hot(matching_classes, depth=3)
        # x-y-w-h-conf-l1-l2 背景，l1, l2 都为0
        # [b, 16, 16, 5, 2]
        matching_classes_one_hot = tf.cast(matching_classes_one_hot[..., 1:], dtype=tf.float32)

        # [b,512,512,3]
        # [b,16,16,5,1]
        # [b,16,16,5,5]
        # [b,16,16,5,2]
        # [b,40,5]
        yield imgs, detector_mask, matching_gt_box, matching_classes_one_hot, gt_boxes_grid

# 2.3 visualize object mask
# train_db -> aug_train_db -> train_gen
train_gen = ground_truth_generator(aug_train_db)

imgs, detector_mask, matching_gt_box, matching_classes_one_hot, gt_boxes_grid =\
    next(train_gen)

img, detector_mask, matching_gt_box, matching_classes_one_hot, gt_boxes_grid =\
    imgs[0], detector_mask[0], matching_gt_box[0], matching_classes_one_hot[0], gt_boxes_grid[0]

flg,(ax1, ax2) = plt.subplots(2, figsize=(5,10))
ax1.imshow(img)
# [16,16,5,1] => [16,16,1]
mask = tf.reduce_sum(detector_mask, axis=2)
print(mask[...,0])
ax2.matshow(mask[...,0]) # [16, 16]

# 3.1
from tensorflow.keras import layers

import tensorflow.keras.backend as K 

class SpaceToDepth(layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                             reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
        return t

    def compute_output_shape(self, input_shape):
        shape =  (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                  input_shape[3] * self.block_size **2)
        return tf.TensorShape(shape)

input_image = layers.Input((IMGSZ, IMGSZ, 3), dtype='float32')

# unit1
# 不使用bias
x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

x = layers.MaxPooling2D(pool_size=(2,2))(x)

# unit2
x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
#  strides: If None, it will default to pool_size.
x = layers.MaxPooling2D(pool_size=(2,2))(x)

# unit3
x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit4
x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_4')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit5
x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_5')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

# unit6
x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_6')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit7
x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_7')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit8
x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_8')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

# unit9
x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_9')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit10
x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_10')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit11
x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_11')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit12
x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_12')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit13
x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_13')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# for skip connection
skip_x = x # [b, 32, 32, 512]

x = layers.MaxPooling2D(pool_size=(2,2))(x)

# unit14
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_14')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit15
x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_15')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit16
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_16')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit17
x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_17')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit18
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_18')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit19
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_19')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit20
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_20')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# unit21
skip_x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(skip_x)
skip_x = layers.BatchNormalization(name='norm_21')(skip_x)
skip_x = layers.LeakyReLU(alpha=0.1)(skip_x)

skip_x = SpaceToDepth(block_size=2)(skip_x)

# concat
# [b, 16, 16, 1024], [b, 16, 16, 256] => [b, 16, 16, 1280]
x = tf.concat([skip_x, x], axis=-1)

# unit22
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_22')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.5)(x) # add dropout

# [b, 16, 16, 5, 7] -> [b, 16, 16, 35]
# 加了dropout，所以Dropout的输出有的变0，因此
# 经过了卷积层后，值不会太大
x = layers.Conv2D(5*7, (1,1), strides=(1,1), name='conv_23', padding='same')(x)

# (5, 7) 35 reshape
output = layers.Reshape((GRIDSZ, GRIDSZ, 5, 7))(x)

# create model
model = keras.models.Model(input_image, output)
x = tf.random.normal((4, 512, 512, 3))
out = model(x)
print('out:', out.shape)

# 3.2 初始化 get_weights set_weights

# 通过名字索引
class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4
weight_reader = WeightReader('yolo.weights')


weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))
    conv_layer.trainable = True

    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))
        norm_layer.trainable = True

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])

layer = model.layers[-2] # last convolutional layer
print(layer.name)

layer.trainable = True
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape) / (GRIDSZ * GRIDSZ)
new_bias = np.random.normal(size=weights[1].shape) / (GRIDSZ * GRIDSZ)

layer.set_weights([new_kernel, new_bias])

# 加载网络训练好的参数
# model.load_weights('ckpt.h5')

# 3.2 网络输出可视化

imgs, detector_mask, matching_gt_boxes, matching_classes_one_hot, gt_boxes_grid =\
    next(train_gen)

img, detector_mask, matching_gt_box, matching_classes_one_hot, gt_boxes_grid =\
    imgs[0], detector_mask[0], matching_gt_boxes[0], matching_classes_one_hot[0], gt_boxes_grid[0]

# [b,512,512,3] => [b, 16, 16, 5, 7] => [16,16,5,x-y-w-h-conf-l1-l2]
# [1,512,512,3]
y_pred = model(tf.expand_dims(img, axis=0))[0][...,4]
# [16, 16, 5] => [16, 16] 那5个锚的iou相加
y_pred = tf.reduce_sum(y_pred, axis=2)

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
ax1.imshow(img)
# [16, 16, 5, 1] => [16, 16]
ax2.matshow(tf.reduce_sum(detector_mask, axis=2)[...,0])
ax3.matshow(y_pred)

# 4.1 coordinate loss

from tensorflow.keras import losses

def compute_iou(x1,y1,w1,h1, x2,y2,w2,h2):
    # x1...: [b,16,16,5]
    xmin1 = x1 - 0.5*w1
    xmax1 = x1 + 0.5*w1
    ymin1 = y1 - 0.5*h1
    ymax1 = y1 + 0.5*h1

    xmin2 = x2 - 0.5*w2
    xmax2 = x2 + 0.5*w2
    ymin2 = y2 - 0.5*h2
    ymax2 = y2 + 0.5*h2

    # (xmin1, ymax1, xmax1, ymin1) (xmin2, ymax2, xmax2, ymin2)
    interw = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    interh = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interw * interh
    union = w1 * h1 + w2 * h2 -inter
    iou = inter / (union + 1e-8)

    # [b,16,16,5]
    return iou

def yolo_loss(detector_mask, matching_gt_boxes, matching_classes_one_hot, gt_boxes_grid, y_pred):
    # detector_mask: [b,16,16,5,1]
    # matching_gt_boxes: [b,16,16,5,5] x-y-w-h-l
    # matching_classes_one_hot: [b,16,16,5,2] l1-l2
    # gt_boxes_grid: [b, 40, 5] x-y-w-h-l
    # y_pred: [b,16,16,5,7] x-y-w-h-conf-l0-l1

    anchors = np.array(ANCHORS).reshape(5,2)

    # creat starting position for each grid anchors
    # [16, 16]
    x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])
    # print(x_grid) shape:(256, )
    # [1,16,16,1,1]
    # [b,16,16,5,2]
    x_grid = tf.reshape(x_grid, (1,GRIDSZ,GRIDSZ,1,1))
    x_grid = tf.cast(x_grid, tf.float32)
    # [b,16_1,16_2,1,1]=>[b,16_2,16_1,1,1]
    y_grid = tf.transpose(x_grid, (0,2,1,3,4))
    xy_grid = tf.concat([x_grid, y_grid], axis=-1)
    # print(xy_grid) 16 * 16 每个点的坐标
    # [1,16,16,1,2]=>[b,16,16,5,2]
    xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1,1,5,1])

    # 得到每个点的绝对位置
    # [b,16,16,5,7] x-y-w-h-conf-l1-l2
    pred_xy = tf.sigmoid(y_pred[..., 0:2])
    pred_xy = pred_xy + xy_grid
    # [b,16,16,5,2]
    pred_wh = tf.exp(y_pred[..., 2:4])
    # [b,16,16,5,2] * [5,2] => [b,16,16,5,2]
    pred_wh = pred_wh * anchors

    # 统计boxes数量
    n_detector_mask = tf.reduce_sum(tf.cast(detector_mask>0., tf.float32))

    # [b,16,16,5,1] * [b,16,16,5,2]
    # 均方差 有object才考虑loss
    xy_loss = detector_mask * tf.square(matching_gt_boxes[...,:2] - pred_xy) / (n_detector_mask+1e-6)
    xy_loss = tf.reduce_sum(xy_loss)

    wh_loss = detector_mask * tf.square(tf.sqrt(matching_gt_boxes[...,2:4])\
         - tf.sqrt(pred_wh)) / (n_detector_mask+1e-6)
    wh_loss = tf.reduce_sum(wh_loss)

    # 4.1 coordinate loss
    coord_loss = xy_loss + wh_loss

    # print(float(coord_loss))
    # 2.729504108428955

    # 4.2 class loss
    # [b,16,16,5,2]
    pred_box_class = y_pred[...,-2:]
    # [b,16,16,5]
    # Returns the index with the largest value across axes of a tensor.
    true_box_class = tf.argmax(matching_classes_one_hot, -1)
    # [b,16,16,5] vs [b,16,16,5,2]
    class_loss = losses.sparse_categorical_crossentropy(\
        true_box_class, pred_box_class, from_logits=True)

    # [b,16,16,5] => [b,16,16,5,1] * [b,16,16,5,1]
    class_loss = tf.expand_dims(class_loss, -1) * detector_mask
    class_loss = tf.reduce_sum(class_loss) / (n_detector_mask + 1e-6)

    # print(float(class_loss))
    # 0.6974056363105774

    # 4.3
    # nonobject_mask
    # iou
    # [b,16,16,5]
    x1,y1,w1,h1 = matching_gt_boxes[...,0],matching_gt_boxes[...,1],\
        matching_gt_boxes[...,2],matching_gt_boxes[...,3]
    x2,y2,w2,h2 = pred_xy[...,0],pred_xy[...,1],\
        pred_wh[...,0],pred_wh[...,1]

    ious = compute_iou(x1,y1,w1,h1, x2,y2,w2,h2)

    # [b,16,16,5,1]
    # 每一个anchor box与它对应的gt box的iou
    ious = tf.expand_dims(ious, axis=-1)

    # [b,16,16,5,1]
    pred_conf = tf.sigmoid(y_pred[...,4:5])

    # 计算每个anchor box 与 每个gt box的iou
    # [b,16,16,5,2] => [b,16,16,5,1,2]
    pred_xy = tf.expand_dims(pred_xy, axis=-2)
    # [b,16,16,5,2] => [b,16,16,5,1,2]
    pred_wh = tf.expand_dims(pred_wh, axis=-2)

    pred_wh_half = pred_wh / 2.
    pred_xymin = pred_xy - pred_wh_half
    pred_xymax = pred_xy + pred_wh_half

    # [b,40,5] => [b,1,1,1,40,5]
    true_boxes_grid = tf.reshape(gt_boxes_grid, \
        [gt_boxes_grid.shape[0], 1, 1, 1, gt_boxes_grid.shape[1], gt_boxes_grid.shape[2]])

    true_xy = true_boxes_grid[...,0:2]
    true_wh = true_boxes_grid[...,2:4]
    true_wh_half = true_wh / 2.
    true_xymin = true_xy - true_wh_half
    true_xymax = true_xy + true_wh_half

    # predxymin, predxymax, true_xymin, true_xymax
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2] => [b,16,16,5,40,2] 
    intersectxymin = tf.maximum(pred_xymin, true_xymin)
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2] => [b,16,16,5,40,2] 
    intersectxymax = tf.minimum(pred_xymax, true_xymax)

    intersect_wh = tf.maximum(intersectxymax - intersectxymin, 0.)

    # [b,16,16,5,40] * [b,16,16,5,40] => [b,16,16,5,40]
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]

    # [b,16,16,5,1,2]
    pred_area = pred_wh[...,0] * pred_wh[...,1]
    # [b,1,1,1,40,2]
    true_area = true_wh[...,0] * true_wh[...,1]

    # [b,16,16,5,1] + [b,1,1,1,40] - [b,16,16,5,40] => [b,16,16,5,40]
    union_area = pred_area + true_area - intersect_area

    # [b,16,16,5,40]
    iou_score = intersect_area / union_area

    # [b,16,16,5,40] => [b,16,16,5] 把无效的gt box也算在40里面了
    best_iou = tf.reduce_max(iou_score, axis=4)

    # [b,16,16,5] => [b,16,16,5,1]
    best_iou = tf.expand_dims(best_iou, axis=-1)

    # 无object的区域设为True,选出来
    nonobj_detection = tf.cast(best_iou<0.6, tf.float32)
    # 在无object区域的基础上,再把有错误的无object区域消除掉. 0*1=0 1*-1=-1(特殊情况,detector_mask=2),所以加下面一句
    nonobj_mask = nonobj_detection * (1 - detector_mask)
    # nonobj counter
    n_nonobj = tf.reduce_sum(tf.cast(nonobj_mask>0., tf.float32))

    # 无object的区域才要计算,所以有nonobj_mask,从下面的obj_loss看,pred_conf上升,而-pred_conf下降,变小,loss减小
    nonobj_loss = tf.reduce_sum(nonobj_mask * tf.square(-pred_conf)) \
        /(n_nonobj + 1e-6)
    # 希望ious与pred_conf接近
    obj_loss = tf.reduce_sum(detector_mask * tf.square(ious - pred_conf))\
        / (n_detector_mask+1e-6)

    # 总结出一个系数5,强调pred_conf逼近ious.
    loss = coord_loss + class_loss + nonobj_loss + 5 * obj_loss

    return loss, [nonobj_loss + 5 * obj_loss, class_loss, coord_loss]

imgs, detector_mask, matching_gt_boxes, matching_classes_one_hot, gt_boxes_grid =\
    next(train_gen)

img, detector_mask, matching_gt_boxes, matching_classes_one_hot, gt_boxes_grid =\
    imgs[0], detector_mask[0], matching_gt_boxes[0], matching_classes_one_hot[0], gt_boxes_grid[0]

y_pred = model(tf.expand_dims(img, axis=0))[0]

loss,sub_loss = yolo_loss(tf.expand_dims(detector_mask,axis=0),
tf.expand_dims(matching_gt_boxes,axis=0),
tf.expand_dims(matching_classes_one_hot,axis=0), 
tf.expand_dims(gt_boxes_grid,axis=0), 
tf.expand_dims(y_pred,axis=0))

# 5.1 train

val_db = get_dataset('data/val/image', 'data/val/annotation', 4)
val_gen = ground_truth_generator(val_db)

def train(epochs):
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9,
        beta_2=0.999, epsilon=1e-8)

    for epoch in range(epochs):

        for step in range(30): # 116 / 4 大约

            img, detector_mask, matching_gt_boxes, matching_classes_one_hot,\
                true_boxes = next(train_gen)
            # print(img.shape)
            with tf.GradientTape() as tape:
                y_pred = model(img, training=True)

                loss, sub_loss = yolo_loss(detector_mask,
                                            matching_gt_boxes,
                                            matching_classes_one_hot,
                                            true_boxes,
                                            y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(epoch, step, float(loss), float(sub_loss[0]), float(sub_loss[1]), float(sub_loss[2]))

# 训练优化
# train(10)
# model.save_weights('weights/epoch10.ckpt')


# 5.2 可视化
model.load_weights('ckpt.h5')
import cv2

def visualize_result(img, model):
    # [512,512,3] 0~255 BGR
    img = cv2.imread(img)
    img = img[...,::-1] / 255.
    # [1,512,512,3]
    img = tf.expand_dims(img, axis=0)
    # [1,16,16,5,7]
    y_pred = model(img, training=False)

    # 构造xy_grid
    x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])
    # [1,16,16,1,1]
    x_grid = tf.reshape(x_grid, (1, GRIDSZ, GRIDSZ, 1, 1))
    x_grid = tf.cast(x_grid, dtype=tf.float32)
    y_grid = tf.transpose(x_grid, (0,2,1,3,4))
    xy_grid = tf.concat([x_grid, y_grid], axis=-1)

    # [1,16,16,5,2]
    xy_grid = tf.tile(xy_grid, [1,1,1,5,1])

    anchors = np.array(ANCHORS).reshape(5,2)
    pred_xy = tf.sigmoid(y_pred[...,0:2])
    pred_xy = pred_xy + xy_grid
    # normalize 0~1
    pred_xy = pred_xy / tf.constant([16.,16.])

    pred_wh = tf.exp(y_pred[..., 2:4])
    pred_wh = pred_wh * anchors
    pred_wh = pred_wh / tf.constant([16., 16.])

    # [1,16,16,5,1]
    pred_conf = tf.sigmoid(y_pred[..., 4:5])
    # l1 l2
    pred_prob = tf.nn.softmax(y_pred[..., 5:])

    pred_xy, pred_wh, pred_conf, pred_prob = \
        pred_xy[0], pred_wh[0], pred_conf[0], pred_prob[0]

    boxes_xymin = pred_xy - 0.5 * pred_wh
    boxes_xymax = pred_xy + 0.5 * pred_wh
    # [16,16,5,2+2]
    boxes = tf.concat((boxes_xymin, boxes_xymax), axis=-1)
    # [16,16,5,2]
    box_score = pred_conf * pred_prob
    # [16,16,5]
    box_class = tf.argmax(box_score, axis=-1)

    # [16,16,5]
    box_class_score = tf.reduce_max(box_score, axis=-1)
    # [16,16,5]
    pred_mask = box_class_score > 0.60
    # [16,16,5,4] => [N,4]
    # 把 box_class_score > 0.45 的盒子选出来
    boxes = tf.boolean_mask(boxes, pred_mask)
    # [16,16,5] => [N]
    # 把 box_class_score > 0.45 盒子的得分选出来
    scores = tf.boolean_mask(box_class_score, pred_mask)

    # [16,16,5] => [N]
    classes = tf.boolean_mask(box_class, pred_mask)

    # 通过掩码把物体的位置，类别，得分选出来

    boxes = boxes * 512.

    # [N] => [n]
    select_idx = tf.image.non_max_suppression(boxes, scores, 10, iou_threshold=0.3)

    boxes = tf.gather(boxes, select_idx)
    scores = tf.gather(scores, select_idx)
    classes = tf.gather(classes, select_idx)

    # plot
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img[0])
    n_boxes = boxes.shape[0]
    ax.set_title('boxes:%d'%n_boxes)

    for i in range(n_boxes):
        x1,y1,x2,y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        label = classes[i].numpy()

        if label ==0:
            # sugarweet
            color = (0,1,0)
        else:
            color = (1,0,0)

        rect = patches.Rectangle((x1.numpy(), y1.numpy()), w.numpy(), h.numpy(),
                                    linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

files = glob.glob('data/val/image/*.png')
for x in files:
    visualize_result(x, model)
