import subprocess


class Streamer:
    def __init__(self, width, height, fps, rtsp_url,
                 preset="veryfast", tune="zerolatency", crf="23"):
        self.width = width
        self.height = height
        self.fps = fps
        self.rtsp_url = rtsp_url

        self.cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", preset,
            "-tune", tune,
            "-crf", crf,
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_url,
        ]

        print("FFmpeg command:", " ".join(self.cmd))
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg pipe closed")
            raise

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass
