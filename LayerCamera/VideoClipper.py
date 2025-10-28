import glob
import os
import re
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip, concatenate_videoclips

class VideoClipper:
    def __init__(self, dir):
        self.dir = dir

    def clip(self, start_time:datetime, duration:int=5) -> str:
        video_list = []

        for f in glob.glob(f"{self.dir}/*.mp4"):
            f = f.replace(f"{self.dir}/", "")
            # match file regex
            g = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})\.(\d{3})\.mp4$", f)
            print(f.replace(self.dir, ""))

            if g is not None:
                d = datetime(int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]), int(g[6]), int(g[7])*1000)
                video_list.append((d, f))

        video_list = sorted(video_list, key=lambda x: x[0])

        i = 0
        found = False

        # find start point
        while i < len(video_list) - 1:
            if video_list[i][0] < start_time and video_list[i+1][0] > start_time:
                found = True
                break
            i += 1

        if not found:
            raise Exception("Cannot find start point")

        composed_list = []

        duration_remained = duration

        # first file
        v = VideoFileClip(f"{self.dir}/{video_list[i][1]}")
        clip_start = (start_time - video_list[i][0]).total_seconds()
        clip_end = v.end if (v.end - clip_start < duration_remained) else clip_start + duration_remained
        print(video_list[i][0], clip_start, clip_end)
        composed_list.append(v.subclip(clip_start))
        duration_remained -= clip_end - clip_start

        # remain files
        while duration_remained > 0 and i < len(video_list):
            v = VideoFileClip(f"{self.dir}/{video_list[i][1]}")
            clip_start = v.start
            clip_end = v.end if (v.end - clip_start < duration_remained) else clip_start + duration_remained
            print(video_list[i][0], clip_start, clip_end)
            composed_list.append(v.subclip(clip_start, clip_end))
            duration_remained -= clip_end - clip_start
            i += 1

        if duration_remained > 0:
            raise Exception("File length not enough")

        # make directory if not exists
        os.makedirs(f"{self.dir}/clips", exist_ok=True)

        output_path = f"{self.dir}/clips/{start_time.strftime('%Y%m%d-%H%M%S-%f')[:-3]}_{duration}.mp4"

        final_clip = concatenate_videoclips(composed_list)
        final_clip.write_videofile(output_path)

        return output_path

if __name__ == "__main__":
    clipper = VideoClipper("./NOL_share_video_for_clip")
    clipper.clip(datetime(2024, 12, 2, 21, 13, 18), 15)