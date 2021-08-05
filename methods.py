import time
import tempfile
import imagehash
import numpy as np
from PIL import Image
from image_similarity_measures.quality_metrics import *


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.fps = -1
        self.frames = -1
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            self.cap.release()
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = int(self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            self.cap.release()
            raise StopIteration
        return img


def downsample(frame):
    max_height = 100
    max_width = 100
    width, height, _ = frame.shape
    if max_height < height or max_width < width:
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame


def nextState(cArray, cID): # Next State Index, Distance and Next State
    for i in range(cID+1, len(cArray)):
        if cArray[i] != cArray[cID]:
            break
    return i, i-cID, cArray[i]


def merge_groups(temporal_groups, frames, minframes):
    sid = 0
    while sid < frames - minframes:
        cid, cdis, cstate = nextState(temporal_groups, sid)
        if cdis < minframes:
            tdis = cdis
            cid2 = cid
            while tdis < minframes:
                cid2, cdis2, cstate2 = nextState(temporal_groups, cid2)
                temporal_groups[sid:cid2] = [1 - cstate for _ in range(cid2 - sid)]
                tdis += cdis2
            temporal_groups[cid2] = cstate
            sid = cid2
        else:
            sid = cid

    temporal_groups[-minframes:] = [temporal_groups[-(minframes + 1)] for _ in range(minframes)]
    return temporal_groups


def diff_hash(videopath):
    i = -1
    tm = []
    tm_d = []
    frame_provider = VideoReader(videopath)
    for frame in frame_provider:
        i += 1
        conversion = cv2.COLOR_BGR2HSV
        frame = cv2.cvtColor(frame, conversion)
        if i == 0:
            tm.append(0)
            tm_d.append(0)
            width, height, _ = frame.shape
            fps = frame_provider.fps
            frames = frame_provider.frames
            frame = downsample(frame)
            prev_frame = frame
            continue
        ch = imagehash.colorhash(Image.fromarray(np.uint8(frame)).convert('RGB'), 5)
        ph = imagehash.colorhash(Image.fromarray(np.uint8(prev_frame)).convert('RGB'), 5)
        tm.append(ch-ph)
        tm_d.append(abs(tm[-1]-tm[-2]))
        frame = downsample(frame)
        prev_frame = frame
    return tm, tm_d, width, height, fps, frames


def diff_inter(videopath):
    i = -1
    tm = []
    tm_d = []
    comparemethod = cv2.HISTCMP_INTERSECT
    frame_provider = VideoReader(videopath)

    for frame in frame_provider:
        i += 1
        conversion = cv2.COLOR_BGR2HSV
        frame = cv2.cvtColor(frame, conversion)
        if i == 0:
            tm.append(0)
            tm_d.append(0)
            width, height, _ = frame.shape
            fps = frame_provider.fps
            frames = frame_provider.frames
            frame = downsample(frame)
            prev_hist = cv2.calcHist([frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
            continue
        frame = downsample(frame)
        curent_hist = cv2.calcHist([frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
        tm.append(cv2.compareHist(curent_hist[0], prev_hist[0], comparemethod))
        tm_d.append(abs(tm[-1]-tm[-2]))
        prev_hist = curent_hist
    tm[0] = tm[1]
    return tm, tm_d, width, height, fps, frames


def diff_lab(videopath):
    i = -1
    tm = []
    tm_d = []
    frame_provider = VideoReader(videopath)
    for frame in frame_provider:
        i += 1
        conversion = cv2.COLOR_BGR2Lab
        frame = cv2.cvtColor(frame, conversion)
        if i == 0:
            tm.append(0)
            tm_d.append(0)
            width, height, _ = frame.shape
            fps = frame_provider.fps
            frames = frame_provider.frames
            frame = downsample(frame)
            prev_frame = frame
            continue
        frame = downsample(frame)
        d = (frame - prev_frame).astype(float)
        mse = np.sqrt(np.einsum('...i,...i->...', d, d))
        tm.append(np.average(mse))
        tm_d.append(abs(tm[-1] - tm[-2]))
        prev_frame = frame
    return tm, tm_d, width, height, fps, frames


def diff_ssim(videopath):
    i = -1
    tm = []
    tm_d = []
    frame_provider = VideoReader(videopath)
    for frame in frame_provider:
        i += 1
        conversion = cv2.COLOR_BGR2HSV
        frame = cv2.cvtColor(frame, conversion)
        if i == 0:
            tm.append(0)
            tm_d.append(0)
            width, height, _ = frame.shape
            fps = frame_provider.fps
            frames = frame_provider.frames
            frame = downsample(frame)
            prev_frame = frame
            continue
        frame = downsample(frame)
        tm.append(ssim(frame, prev_frame))
        tm_d.append(abs(tm[-1]-tm[-2]))
        prev_frame = frame
    return tm, tm_d, width, height, fps, frames


def diff_other(videopath):
    i = -1
    tm = []
    tm_d = []
    frame_provider = VideoReader(videopath)
    for frame in frame_provider:
        i += 1
        conversion = cv2.COLOR_BGR2HSV
        frame = cv2.cvtColor(frame, conversion)
        if i == 0:
            prev_frame = frame
            tm.append(0)
            tm_d.append(0)
            width, height, _ = frame.shape
            fps = frame_provider.fps
            frames = frame_provider.frames
            continue
        # METHOD GOES HERE
        tm.append()
        tm_d.append(abs(tm[-1]-tm[-2]))
        prev_frame = frame
    return tm, tm_d, width, height, fps, frames


def group_extractor(input, groups, height, width, fps, output):
    video = VideoReader(input)

    whitescreen = np.zeros([height, width, 3], dtype=np.uint8)
    whitescreen.fill(255)

    i = -1
    groupID = 0
    prev_group = groups[0]
    fourcc = cv2.VideoWriter_fourcc('V', 'P', '8', '0')
    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tempVideo = tempfile.NamedTemporaryFile(suffix='.webM')
    #tempVideo.name
    writer = cv2.VideoWriter(tempVideo.name, fourcc, fps, (width, height))

    for frame in video:
        i += 1
        if prev_group == groups[i]: # No group change
            writer.write(frame)
        else: # Group change
            groupID += 1
            for ws in range(fps):
                writer.write(whitescreen)
            writer.write(frame)
        prev_group = groups[i]
    writer.release()
    return tempVideo.name


def method(video, threshold, minframes, method):
    if method == "Hash":
        difference_check = diff_hash
    elif method == "Intersection":
        difference_check = diff_inter
    elif method == "Lab":
        difference_check = diff_lab
    elif method == "SSIM":
        difference_check = diff_ssim
    else:
        difference_check = diff_other

    start_time = time.time()
    differences, sdifferences, width, height, fps, frames = difference_check("./source/" + video)
    method_time = time.time() - start_time

    start_time = time.time()
    norm_tm = np.array(differences) * 100 / max(differences)
    norm_tm_diff = np.array(sdifferences) * 100 / max(sdifferences)
    groups = np.zeros(len(norm_tm))
    td_indices = norm_tm > threshold
    groups[td_indices] = 1
    groups = merge_groups(groups, frames, minframes)
    merge_time = time.time() - start_time

    start_time = time.time()
    output = group_extractor("./source/" + video, groups, width, height, fps, "./result/"+video.rstrip('.mp4/')+".webM")
    extract_time = time.time() - start_time

    return norm_tm, groups, norm_tm_diff, (frames/fps, method_time, merge_time, extract_time), output

#_, _, _, _ = method("10000000_769221737104889_6896866277500076481_n.mp4", 80, 15, "Lab")
