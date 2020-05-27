from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np

def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def draw_rect(img_ori, boxes=None, name = '1.jpg' , box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    img = img_ori.copy()
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['centerhw' , 'ltwh', 'ltrb']  #'centerwh' : (center_y,center_x,h,w) , ltwh : top-left + w +h ,  ltrb : top-left + bottom-right
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        if box_fmt == 'centerhw':
            box = boxes.copy()
            box[:,0] = boxes[:,1] - 1 - (box[:,3] - 1) / 2
            box[:,1] = boxes[:,0] - 1 - (box[:,2] - 1) / 2
            box[:,2] = boxes[:,3]
            box[:,3] = boxes[:,2]
            boxes = box
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
        
    cv2.imwrite( name , img)
    return img

def Tensor_to_CV(tensor):
    img = tensor.byte().cpu().numpy()
    try : 
        img = img.squeeze(0)
    except:
        img = img.copy()
    img = img.transpose((1, 2, 0))
    return img

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR,
                    need_gt = False , 
                    gt = None):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    if need_gt == True:
        gt_new = gt.copy()
        gt_new[0] = 0
        gt_new[1] = 0
        gt_new[2:] = gt[2:] / size * out_size
        # gt_new[:2] = np.array([(out_size-1)/2 , (out_size-1)/2])
        return patch, gt_new
    return patch

def create_anchors(cfg, response_sz):
    anchor_num = len(cfg.ratios) * len(cfg.scales)
    anchors = np.zeros((anchor_num, 4), dtype=np.float32)

    size = cfg.total_stride * cfg.total_stride
    ind = 0
    for ratio in cfg.ratios:
        w = int(np.sqrt(size / ratio))
        h = int(w * ratio)
        for scale in cfg.scales:
            anchors[ind, 0] = 0
            anchors[ind, 1] = 0
            anchors[ind, 2] = w * scale
            anchors[ind, 3] = h * scale
            ind += 1
    anchors = np.tile(
        anchors, response_sz * response_sz).reshape((-1, 4))

    begin = -(response_sz // 2) * cfg.total_stride
    xs, ys = np.meshgrid(
        begin + cfg.total_stride * np.arange(response_sz),
        begin + cfg.total_stride * np.arange(response_sz))
    xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
    ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
    anchors[:, 0] = xs.astype(np.float32)
    anchors[:, 1] = ys.astype(np.float32)

    return anchors

def compute_target(anchors, box):
    regression_target = box_transform(anchors, box)#box=[gt_cx,gt_cy,gt_w,gt_h]，regression—target 回归值offset
                        
    iou = compute_iou(anchors, box).flatten()#1805个iou
    # print(np.max(iou))
    pos_index = np.where(iou > 0.6)[0] #返回大于0.6的index，作为正样本索引
    neg_index = np.where(iou < 0.3)[0] #返回小于0.3的index，作为负样本索引
    label = np.ones_like(iou , dtype = np.int64) * -1 #大于0.6等于1； 小于0.3等于0； 介于0.3和0.6之间的数等于-1
    label[pos_index] = 1
    label[neg_index] = 0
    return regression_target, label

def box_transform(anchors, gt_box):
    anchor_xctr = anchors[:, :1]  #cx
    anchor_yctr = anchors[:, 1:2] #cy
    anchor_w = anchors[:, 2:3]   
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w # offset-x
    target_y = (gt_cy - anchor_yctr) / anchor_h # offset-y
    target_w = np.log(gt_w / anchor_w) #offset-w
    target_h = np.log(gt_h / anchor_h) #offset-h
    regression_target = np.hstack((target_x, target_y, target_w, target_h)) #hstack((1805,1),(1805,1),(1805,1),(1805,1))左右拼接
    return regression_target

def compute_iou(anchors, box):
    # print(box)
    # print(anchors[0])
    if np.array(anchors).ndim == 1:#几个维度
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:    #几个维度
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))#重复数组来构建新的数组[1805,4]

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5 # cx-(w-1)/2=x1
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5 # cx+(w-1)/2=x2
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5 # cy-(h-1)/2=y1
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5 # cy+(h-1)/2=y2

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)
    #计算相交的区域
    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou