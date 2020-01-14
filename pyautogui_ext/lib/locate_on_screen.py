import os
import datetime

import fire
from pyscreeze import screenshot

import numpy as np
import cv2

# this is a trick to cheat PyCharm to support autocomplete
try:
    import cv2.cv2 as cv2
except Exception:
    pass

from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__file__)

# ref: https://stackoverflow.com/questions/43390654/opencv-3-2-nameerror-global-name-flann-index-lsh-is-not-defined
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6


def _temp_file(temp_dir, prefix='', suffix=''):
    os.makedirs(temp_dir, exist_ok=True)
    return '{}{}{}'.format(prefix, datetime.datetime.now().strftime('%Y-%m%d_%H-%M-%S-%f'), suffix)


def _pil_to_cv2_image(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2GRAY)  # 从一种颜色空间转换为另一种颜色空间


def _create_default_flann_matcher():
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _create_default_bf_matcher():
    return cv2.BFMatcher()


def feature_match(img1, img2, detector=None, matcher=None, ratio_test=0.75, min_matches=4, debug=True,
                  temp_dir='./tmp'):
    if detector is None:
        detector = cv2.xfeatures2d.SURF_create(800)
    if matcher is None:
        matcher = _create_default_bf_matcher()

    kp1, des1 = detector.detectAndCompute(img1, None)  # 检测图像的关键点及其描述信息
    kp2, des2 = detector.detectAndCompute(img2, None)  # 检测图像的关键点及其描述信息

    matches = matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)  # 返回k个最佳匹配

    # select good matches via ratio test as per Lowe's paper
    good_matches = []
    for i, match in enumerate(matches):
        if 2 == len(match):
            m, n = match
            if m.distance < ratio_test * n.distance:
                good_matches.append(match)
        elif 1 == len(match):  # there are chances that only 1 match point is found
            good_matches.append(match)

    logger.info('{} good matches are found (ratio: {})'.format(len(good_matches), ratio_test))
    assert len(good_matches) >= min_matches, \
        'good match points is less than min_matches: {}'.format(min_matches)

    # find homography matrix
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    trans, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 查找两个平面之间的透视转换。
    matches_mask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, trans)  # 执行向量的透视矩阵转换。
    x1, y1 = dst[0][0]
    x2, y2 = dst[2][0]
    x = (x1 + x2) / 4
    y = (y1 + y2) / 4
    return x, y

    # ui.moveTo(x, y)

    # for debug
    # img2 = cv2.polylines(img2, [np.int32(dst)], True, 128, 9, cv2.LINE_AA)  # 绘制多条多边形曲线
    #
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=None,
    #                    matchesMask=matches_mask,
    #                    flags=2)
    #
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, [m[0] for m in good_matches], None, **draw_params)
    # plt.imshow(img3, ), plt.show()


class AutoGui:
    def __init__(self, debug=False):
        self._debug = debug

    def locate_on_screen(self, ref: str):
        # load reference image
        ref_img = cv2.cvtColor(cv2.imread(ref), cv2.COLOR_BGR2GRAY)

        # take screen shot
        screenshot_filename = None
        if self._debug:
            screenshot_filename = 'screenshot%s.png' % (datetime.datetime.now().strftime('%Y-%m%d_%H-%M-%S-%f'))
            logger.info('screenshot file {}'.format(screenshot_filename))

        sensed_img = _pil_to_cv2_image(screenshot(screenshot_filename, region=None))
        return feature_match(ref_img, sensed_img)


if __name__ == '__main__':
    fire.Fire(AutoGui)
