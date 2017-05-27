import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class imagenet(imdb):
    def __init__(self, image_set, year, data_path=None):
        imdb.__init__(self, 'imagenet_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._data_path = data_path if data_path else self._get_default_path()
        self._classes = ("__background__", # always index 0
            "n02672831", "n02691156", "n02219486", "n02419796", "n07739125", 
            "n02454379", "n07718747", "n02764044", "n02766320", "n02769748", 
            "n07693725", "n02777292", "n07753592", "n02786058", "n02787622", 
            "n02799071", "n02802426", "n02807133", "n02815834", "n02131653", 
            "n02206856", "n07720875", "n02828884", "n02834778", "n02840245", 
            "n01503061", "n02870880", "n02883205", "n02879718", "n02880940", 
            "n02892767", "n07880968", "n02924116", "n02274259", "n02437136", 
            "n02951585", "n02958343", "n02970849", "n02402425", "n02992211", 
            "n01784675", "n03000684", "n03001627", "n03017168", "n03062245", 
            "n03063338", "n03085013", "n03793489", "n03109150", "n03128519", 
            "n03134739", "n03141823", "n07718472", "n03797390", "n03188531", 
            "n03196217", "n03207941", "n02084071", "n02121808", "n02268443", 
            "n03249569", "n03255030", "n03271574", "n02503517", "n03314780", 
            "n07753113", "n03337140", "n03991062", "n03372029", "n02118333", 
            "n03394916", "n01639765", "n03400231", "n02510455", "n01443537", 
            "n03445777", "n03445924", "n07583066", "n03467517", "n03483316", 
            "n03476991", "n07697100", "n03481172", "n02342885", "n03494278", 
            "n03495258", "n03124170", "n07714571", "n03513137", "n02398521", 
            "n03535780", "n02374451", "n07697537", "n03584254", "n01990800", 
            "n01910747", "n01882714", "n03633091", "n02165456", "n03636649", 
            "n03642806", "n07749582", "n02129165", "n03676483", "n01674464", 
            "n01982650", "n03710721", "n03720891", "n03759954", "n03761084", 
            "n03764736", "n03770439", "n02484322", "n03790512", "n07734744", 
            "n03804744", "n03814639", "n03838899", "n07747607", "n02444819", 
            "n03908618", "n03908714", "n03916031", "n00007846", "n03928116", 
            "n07753275", "n03942813", "n03950228", "n07873807", "n03958227", 
            "n03961711", "n07768694", "n07615774", "n02346627", "n03995372", 
            "n07695742", "n04004767", "n04019541", "n04023962", "n04026417", 
            "n02324045", "n04039381", "n01495701", "n02509815", "n04070727", 
            "n04074963", "n04116512", "n04118538", "n04118776", "n04131690", 
            "n04141076", "n01770393", "n04154565", "n02076196", "n02411705", 
            "n04228054", "n02445715", "n01944390", "n01726692", "n04252077", 
            "n04252225", "n04254120", "n04254680", "n04256520", "n04270147", 
            "n02355227", "n02317335", "n04317175", "n04330267", "n04332243", 
            "n07745940", "n04336792", "n04356056", "n04371430", "n02395003", 
            "n04376876", "n04379243", "n04392985", "n04409515", "n01776313", 
            "n04591157", "n02129604", "n04442312", "n06874185", "n04468005", 
            "n04487394", "n03110669", "n01662784", "n03211117", "n04509417", 
            "n04517823", "n04536866", "n04540053", "n04542943", "n04554684", 
            "n04557648", "n04530566", "n02062744", "n04591713", "n02391049")

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        # self._salt = str(uuid.uuid4())
        # self._comp_id = 'comp4'

        # ImageNet specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        # assert os.path.exists(self._devkit_path), \
        #         'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'ImageNet' + self._year)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def get_real_image_set(self, index):
        if "train" in index:
            image_set="train"
        elif "val" in index:
            image_set="val"
        else:
            assert False, "index should include either train or val"
        return image_set

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_set=self.get_real_image_set(index)
        image_path = os.path.join(self._data_path, 'Data', "DET", 
            image_set, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.split(" ")[0].strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'val2':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        rpn_dir = self.config['rpn_file']
        print 'loading {}'.format(rpn_dir)
        assert os.path.exists(rpn_dir), \
               'rpn data not found at: {}'.format(rpn_dir)
        filenames = os.listdir(rpn_dir)
        box_list = [[] for _ in xrange(len(filenames))]
        count = 0
        for fn in filenames:
            i = int(fn.split(".")[0])
            with open(os.path.join(rpn_dir, fn), "rb") as fp:
                box_list[i] = cPickle.load(fp)
            count += 1
            if count % 1000 == 0:
                print "{}/{}".format(count, len(filenames))
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        real_image_set=self.get_real_image_set(index)
        filename = os.path.join(self._data_path, 'Annotations', 'DET', 
            real_image_set, index + '.xml')
        tree = ET.parse(filename)
        size = tree.find('size')
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if obj.find('difficult')==None or int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            x1, x2 = max(min(x1, x2), 0), min(max(x1, x2), width-1)
            y1, y2 = max(min(y1, y2), 0), min(max(y1, y2), height-1)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    # def _get_voc_results_file_template(self):
    #     # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    #     filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #     path = os.path.join(
    #         self._devkit_path,
    #         'results',
    #         'VOC' + self._year,
    #         'Main',
    #         filename)
    #     return path

    # def _write_voc_results_file(self, all_boxes):
    #     for cls_ind, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         print 'Writing {} VOC results file'.format(cls)
    #         filename = self._get_voc_results_file_template().format(cls)
    #         with open(filename, 'wt') as f:
    #             for im_ind, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in xrange(dets.shape[0]):
    #                     f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                             format(index, dets[k, -1],
    #                                    dets[k, 0] + 1, dets[k, 1] + 1,
    #                                    dets[k, 2] + 1, dets[k, 3] + 1))

    # def _do_python_eval(self, output_dir = 'output'):
    #     annopath = os.path.join(
    #         self._devkit_path,
    #         'VOC' + self._year,
    #         'Annotations',
    #         '{:s}.xml')
    #     imagesetfile = os.path.join(
    #         self._devkit_path,
    #         'VOC' + self._year,
    #         'ImageSets',
    #         'Main',
    #         self._image_set + '.txt')
    #     cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    #     aps = []
    #     # The PASCAL VOC metric changed in 2010
    #     use_07_metric = True if int(self._year) < 2010 else False
    #     print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
    #     if not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     for i, cls in enumerate(self._classes):
    #         if cls == '__background__':
    #             continue
    #         filename = self._get_voc_results_file_template().format(cls)
    #         rec, prec, ap = voc_eval(
    #             filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
    #             use_07_metric=use_07_metric)
    #         aps += [ap]
    #         print('AP for {} = {:.4f}'.format(cls, ap))
    #         with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
    #             cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #     print('Mean AP = {:.4f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('Results:')
    #     for ap in aps:
    #         print('{:.3f}'.format(ap))
    #     print('{:.3f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('')
    #     print('--------------------------------------------------------------')
    #     print('Results computed with the **unofficial** Python eval code.')
    #     print('Results should be very close to the official MATLAB eval code.')
    #     print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #     print('-- Thanks, The Management')
    #     print('--------------------------------------------------------------')

    # def _do_matlab_eval(self, output_dir='output'):
    #     print '-----------------------------------------------------'
    #     print 'Computing results with the official MATLAB eval code.'
    #     print '-----------------------------------------------------'
    #     path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    #            .format(self._devkit_path, self._get_comp_id(),
    #                    self._image_set, output_dir)
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    # def evaluate_detections(self, all_boxes, output_dir):
    #     self._write_voc_results_file(all_boxes)
    #     self._do_python_eval(output_dir)
    #     if self.config['matlab_eval']:
    #         self._do_matlab_eval(output_dir)
    #     if self.config['cleanup']:
    #         for cls in self._classes:
    #             if cls == '__background__':
    #                 continue
    #             filename = self._get_voc_results_file_template().format(cls)
    #             os.remove(filename)

    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
