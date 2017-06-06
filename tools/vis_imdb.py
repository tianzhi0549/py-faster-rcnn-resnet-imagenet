import _init_paths, cv2
from datasets.factory import get_imdb, list_imdbs
import numpy as np

def draw_boxes(im, bboxes, is_display=True, color=None, caption="Image", wait=True):
    """
        boxes: bounding boxes
    """
    im=im.copy()
    for box in bboxes:
        if color==None:
            if len(box)==5 or len(box)==9:
                c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
            else:
                c=tuple(np.random.randint(0, 256, 3))
        else:
            c=color
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c)
    if is_display:
        cv2.imshow(caption, im)
        if wait:
            cv2.waitKey(0)
    return im

imdb=get_imdb("imagenet_2015_trainval1_woextra")
roidb=imdb.gt_roidb()

for i in np.random.permutation(np.arange(imdb.num_images)):
    print imdb.image_path_at(i)
    im=cv2.imread(imdb.image_path_at(i))
    draw_boxes(im, roidb[i]["boxes"])
