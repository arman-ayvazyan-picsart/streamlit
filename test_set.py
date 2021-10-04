from os import listdir, mkdir
from os.path import isfile, join
from methods import video_file as method


source_root = "../FrameGroupping_Archive/source/"
restult_root = "../FrameGroupping_Archive/result/"
folder = "VideoSet2/Diverse light conditions/"
video_files = [f for f in listdir(source_root + folder) if isfile(join(source_root, folder, f))]

print("Processing the subfolder + ", folder)
for file in video_files:
    for m in ("SSIM", "Lab"):
        for a in (True, False):
            print(str(file), "-", m, str(a))
            method(file, source_root + folder, restult_root + folder, m, a, 60, True, False)
