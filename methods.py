import time
import argparse
import matplotlib.pyplot as plt
from image_similarity_measures.quality_metrics import *


class VideoReader(object):
    """
    VideoReader is a class to read frames from a video file,
        while providing some useful metadata about the video.

    :param file_name: Video name
    """
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


def downsample(frame, max_height=256, max_width=256):
    """
    downsample is a helper function used to improve performance by
        downsampling (keeping the ratio) frames
        before feeding into similarity method.

    :param frame: Original frame
    :param max_height:(optional) Maximum height of the frame
    :param max_width:(optional) Maximum width of the frame
    :return: downsampled frame
    """
    width, height, _ = frame.shape
    if max_height < height or max_width < width:
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame


def next_state(carray, cid):  # Next State Index, Distance and Next State
    """
    next_state is a helper function used while merging the small groups.

    :param carray: array to look in
    :param cid: index of the current possition
    :return: next state index, distance to the next state, next state value
    """
    i = len(carray)-1
    for i in range(cid+1, len(carray)):
        if carray[i] != carray[cid]:
            break
    return i, i-cid, carray[i]


def merge_groups(groups, frames, minframes):
    """
    merge_groups is merges small cut clips into bigger
        ones. Anything smaller than minframes will be
        joined with the next group until frames in the
        group are more than minframes

    :param groups: array to look in
    :param frames: index of the current possition
    :param minframes: index of the current possition
    :return: array with value 100 for no-cuts, and lower value for cut points
    """
    sid = 0
    while sid < frames - minframes:
        cid, cdis, cstate = next_state(groups, sid)
        if cdis < minframes:
            tdis = cdis
            cid2 = cid
            while tdis < minframes:
                cid2, cdis2, cstate2 = next_state(groups, cid2)
                groups[sid:cid2] = 100
                tdis += cdis2
            groups[cid2] = 0
            sid = cid2 + 1
        else:
            sid = cid + 1
    groups[-minframes:] = 100
    return groups


def lab(current_frame, prev_frame):
    """
    lab is a similarity comparision method.

    :param current_frame: first frame to compare
    :param prev_frame: second frame to compare
    :return: how similar are the frames
    """
    d = (current_frame - prev_frame).astype(float)
    mse = np.sqrt(np.einsum('...i,...i->...', d, d))
    return np.average(mse)


def compute_similarity(videopath, conversion, method):
    """
    compute_similarity finds how similar are the frames in the video.

    :param videopath: Path to video file
    :param conversion: CV2 color space conversion
    :param method: Reference to a method function with function_name(frame1, frame2) format
    :return: array with frame difference values and several metadata about video
    """

    frame_diff = []
    width, height, fps, frames, prev_frame = 0, 0, 0, 0, 0
    frame_provider = VideoReader(videopath)

    for i, frame in enumerate(frame_provider):
        frame = cv2.cvtColor(frame, conversion)
        if i == 0:  # Different operations for the first frame (as no previous frame is present)
            frame_diff.append(0)
            width, height, _ = frame.shape
            fps = frame_provider.fps
            frames = frame_provider.frames
            frame = downsample(frame)
            prev_frame = frame
            continue
        frame = downsample(frame)
        frame_diff.append(method(frame, prev_frame))
        prev_frame = frame
    return frame_diff, width, height, fps, frames


def adaptive_threshold(frame_diff, ws=60):
    """
    adaptive_threshold is a thresholding function that find better cuts than continues threshold.

    :param frame_diff: array with frame differences
    :param ws: sliding window size
    :return: cut points in the video
    """
    # Defining variables

    peaks = np.ones(frame_diff.shape) * 100
    peaks_index = []
    v_threshold = 1  # To introduce additional control parameter in future (i.e. 0.98, 0.99)

    # Getting the peaks in the array

    max_peak = max(frame_diff)
    min_peak = min(frame_diff)
    mean = (max_peak+min_peak) / 2  # Can be added ACP in future. Mean = max_peak * ACP
    tmp = frame_diff < mean
    peaks[tmp] = frame_diff[tmp]

    # Finding additional peaks in sliding window

    for i in range(ws // 2, len(frame_diff) - ws // 2):
        cw_left = np.average(frame_diff[i - ws // 2:i])
        cw_left *= v_threshold
        cw_right = np.average(frame_diff[i + 1: i + 1 + ws // 2])
        cw_right *= v_threshold
        if frame_diff[i] < cw_left and frame_diff[i] < cw_right:
            peaks[i] = 50
            peaks_index.append(i)

    # Decrease the window size against overlays
    if ws > 35:
        ws -= 20
    peaks_index_clean = []

    # Choose the most different peak in sliding window

    for i in range(ws, len(frame_diff), ws + 10):
        lb = i - ws
        ub = i + ws
        pis = []
        pis_id = []
        for pid, item in enumerate(peaks_index):
            if (item > lb) and (item < ub):
                pis.append(frame_diff[item])
                pis_id.append(item)
            else:
                pass
        if len(pis) > 0:
            pis_min_id = np.argmin(pis)
            peaks_index_clean.append(pis_id[pis_min_id])

    # Clean up duplicate peaks and sort
    peaks_index = list(set(peaks_index_clean))
    peaks_index.sort()

    # Prepare the output with real values

    peaks = np.ones(frame_diff.shape) * 100
    for i in range(len(peaks_index)):
        peaks[peaks_index[i]] = frame_diff[peaks_index[i]]

    return peaks


def group_extractor(input_video, groups, height, width, fps, output):
    """
    group_extractor inserts 1 second whitescreens into cut points and exports a video.

    :param input_video: Input video name
    :param groups: Cut points
    :param height: Video frame height
    :param width: Video frame width
    :param fps: Video FPS
    :param output: Output path + new video name
    :return: returns nothing
    """
    video = VideoReader(input_video)
    whitescreen = np.zeros([height, width, 3], dtype=np.uint8)
    whitescreen.fill(255)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    for i, frame in enumerate(video):
        if groups[i] == 100:
            writer.write(frame)
        else:
            for ws in range(fps):
                writer.write(whitescreen)
            writer.write(frame)
    writer.release()


def plot_results(main_array, output_folder, videoname, plot_groups=True, overlay=0):
    """
    plot_results is a function to save charts in case verbose mode is on.
    :param main_array: Array to plot
    :param output_folder: Output folder path
    :param videoname: Video name to include in the chart name
    :param plot_groups: Boolean to either plot groups or frame similarities
    :param overlay: Additional array or threshold value to show on chart (in case plot_groups=False)
    :return: nothing
    """
    plt.clf()
    plot_title = "Frame Groups" if plot_groups else "Frame Differences"
    plt.title(plot_title)
    if plot_groups:
        frames_data = np.arange(0, len(main_array))
        plt.step(frames_data, main_array)
    elif type(overlay) == np.float64:
        plt.plot(main_array, 'ro')
        plt.axhline(y=overlay, color='r', linestyle='-')
    else:
        plt.plot(main_array, color='b')
        # plt.plot(overlay, color='r')
    plt.savefig(output_folder + videoname + str(plot_title) + " - " + ".jpg")


def video_file(video, subfolder, method_name, adaptive, window_size, video_output, verbose):
    """
    video_file combines all the method to extract cut points from a video FILE.
        To use in streamlit use the video_stream method.

    :param video: Video name
    :param subfolder: To lookup the video in
    :param method_name: Use ssim by default, but if needed can be changed to lab
    :param adaptive: If true, adaptive thresholding will be used, otherwise minmax thresholding
    :param window_size: sliding window size
    :param video_output: If true, a video file will be created with whitescreens,
        otherwise cut points will be printed out
    :param verbose: Show details (duration each section runs and charts)
    :return: nothing
    """

    # Similarity method inference

    start_time = time.time()

    if method_name == "Lab":
        conversion = cv2.COLOR_BGR2Lab
        method = lab
    else:
        conversion = cv2.COLOR_BGR2HSV
        method = ssim

    frame_diff, width, height, fps, frames = compute_similarity("./source/" + subfolder + video, conversion, method)
    method_time = time.time() - start_time

    # Normalizing the array and fixing the first 2 elements
    frame_diff = np.array(frame_diff) * 100 / max(frame_diff)
    frame_diff[0:2] = frame_diff[3]
    groups = np.zeros(len(frame_diff))

    # Thresholding with corresponing method
    if not adaptive:
        start_time = time.time()
        max_peak = max(frame_diff)
        min_peak = min(frame_diff)
        threshold = (max_peak + min_peak) / 2
        overlay = threshold  # For plotting the chart
        td_indices = frame_diff > threshold
        groups[td_indices] = 100
        groups = merge_groups(groups, len(frame_diff), window_size)
        threhsold_time = time.time() - start_time
    else:
        start_time = time.time()
        groups = adaptive_threshold(frame_diff, window_size)
        overlay = groups  # For plotting the chart
        threhsold_time = time.time() - start_time

    # Outputs and details

    suffix = "_ADAPTIVE" if adaptive else "_MINMAX"
    if video_output:
        start_time = time.time()
        group_extractor("./source/" + subfolder + video, groups, width, height, fps,
                        "./result/" + subfolder + video.rstrip('.mp4/') + suffix + ".mp4")
        extract_time = time.time() - start_time
    else:
        extract_time = 0
        frame_ids = [i for i, j in enumerate(groups) if j != 100]
        print("Frame IDs are ", frame_ids)

    if verbose:
        print("Frame similarity method: ", method_name)
        print("Thresholding method: ", suffix.lstrip("_").capitalize())
        print("Video duration: ", round(frames / fps, 3), "s.")
        print("Method duration: ", round(method_time, 3), "s.")
        print("Thresholding duration: ", round(threhsold_time, 3), "s.")
        print("Extraction duration: ", round(extract_time, 3), "s.")
        plot_results(groups, "./result/" + subfolder, video.rstrip('.mp4') + suffix, True)
        plot_results(frame_diff, "./result/" + subfolder, video.rstrip('.mp4') + suffix, False, overlay)


def video_stream(video, method_name, adaptive, window_size):
    """
    video_stream is used in streamlit app to provide entry point into program.

    :param video: Video name
    :param method_name: Use ssim by default, but if needed can be changed to lab
    :param adaptive: If true, adaptive thresholding will be used, otherwise minmax thresholding
    :param window_size: sliding window size
    :return: nothing
    """

    # Similarity method inference

    start_time = time.time()

    if method_name == "Lab":
        conversion = cv2.COLOR_BGR2Lab
        method = lab
    else:
        conversion = cv2.COLOR_BGR2HSV
        method = ssim

    frame_diff, width, height, fps, frames = compute_similarity(video, conversion, method)
    method_time = time.time() - start_time

    # Normalizing the array and fixing the first 2 elements
    frame_diff = np.array(frame_diff) * 100 / max(frame_diff)
    frame_diff[0:2] = frame_diff[3]
    groups = np.zeros(len(frame_diff))

    # Thresholding with corresponing method
    if not adaptive:
        start_time = time.time()
        max_peak = max(frame_diff)
        min_peak = min(frame_diff)
        threshold = (max_peak + min_peak) / 2
        overlay = threshold  # For plotting the chart
        td_indices = frame_diff > threshold
        groups[td_indices] = 100
        groups = merge_groups(groups, len(frame_diff), window_size)
        threhsold_time = time.time() - start_time
    else:
        start_time = time.time()
        groups = adaptive_threshold(frame_diff, window_size)
        overlay = groups  # For plotting the chart
        threhsold_time = time.time() - start_time

    # Outputs and details
    frame_ids = [i for i, j in enumerate(groups) if j != 100]
    video_duration = round(frames / fps, 3)
    method_duration = round(method_time, 3)
    thresholding_duration = round(threhsold_time, 3)

    return frame_ids, (video_duration, method_duration, thresholding_duration), frame_diff, groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adaptive-threshold', type=bool, default=True,
                        help='Thresholding method. Use True for Adaptive and False  for MinMax')
    parser.add_argument('-method-name', type=str, default="ssim", help='Use method either Lab or ssim')
    parser.add_argument('-window-size', type=int, default=60,
                        help='Sliding window size used for merging short clips into bigger ones')
    parser.add_argument('-verbose', type=bool, default=True,
                        help='Show timings and plot charts')
    parser.add_argument('-video-output', type=bool, default=True,
                        help='Export a video in the corresponding folder. If False only frameIDs will be shown')
    args = parser.parse_args()
    video_file("How to use Jade Roller_ Directions_Steps to Use_.mp4",
               "VideoSet/tutorial/tutorial - continuous flow/", args.method_name,
               args.adaptive_threshold, args.window_size, args.video_output, args.verbose)
    #  video_stream()
