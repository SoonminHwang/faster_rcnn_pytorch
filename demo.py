import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize(image, bbox, scores, names, classes, save_name=None):
    cls_to_ind = { cls:ind for ind, cls in enumerate(classes) }

    clrs = sns.color_palette("Set2", len(classes))
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    im2show = np.copy(image.astype(np.uint8))
    plt.imshow(im2show[:,:,(2,1,0)])
    axe = plt.gca()
    bLabel = np.zeros(len(classes))
    for det, score, objName in zip(bbox, scores, names):
        det = tuple(int(x) for x in det)
        clr = clrs[ cls_to_ind[objName] ]
        if bLabel[ cls_to_ind[objName] ] == 0:
            rect = plt.Rectangle( (det[0], det[1]), det[2]-det[0], det[3]-det[1], 
                fill=False, edgecolor=clr, linewidth=1.5, label=objName)
            bLabel[ cls_to_ind[objName] ] = 1
        else:
            rect = plt.Rectangle( (det[0], det[1]), det[2]-det[0], det[3]-det[1], 
                fill=False, edgecolor=clr, linewidth=1.5)            

        axe.add_patch(rect)
        axe.text(det[0], det[1]-2, '%.3f'%score,
            bbox=dict(facecolor=clr, alpha=0.5), fontsize=7, color='white')
       
    if any(bLabel): 
        leg = plt.legend(loc='best')
        for text in leg.get_texts():
            plt.setp(text, color = 'w')

    plt.tight_layout()

    if not save_name:
        save_name = os.path.join('demo', 'out.jpg')
    plt.savefig(save_name, dpi=200)


def test():
    
    # im_file = 'demo/004545.jpg'
    # im_file = 'demo/000013.png'
    # im_file = 'demo/000068.png'
    # im_file = 'demo/000035.jpg'
    im_file = 'demo/000007.png'
        
    image = cv2.imread(im_file)

    #model_file = 'models/saved_model3/faster_rcnn_100000.h5'
    model_file = 'outputs/kitti_vgg16_rgb/kittivoc_2017-04-04_06-00/faster_rcnn_10000.h5'

    classes = ('__background__', 'Pedestrian', 'Car', 'Cyclist')
    detector = FasterRCNN(classes=classes)    

    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    detector.MAX_SIZE = 3000
    print 'Detector scale: %d ->' % detector.SCALES[0],
    detector.SCALES = (600,)
    print '%d.' % detector.SCALES[0]

    dummy = np.zeros((755, 2500, 3), dtype=np.uint8) + 128    
    for ii in xrange(10):
        detector.detect(dummy, 0.7)

    t = Timer()
    t.tic()    
    dets, scores, clsNames = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    visualize(image, dets, scores, clsNames, classes, im_file.replace('/', '/[out]'))


if __name__ == '__main__':
    test()
