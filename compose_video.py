import os
import subprocess

# ================= CẤU HÌNH ĐƯỜNG DẪN INPUT / OUTPUT =================
SPECTRUM_MOV_PATH = "Output_audio/acv_card_spectrum.mov"   # File sóng nhạc + ảnh nền trong suốt
BG_VIDEO_PATH = "Input_background/background.mp4"          # Video nền
AUDIO_PATH = "Input_audio/audio.WAV"                       # File nhạc gốc chất lượng cao
FINAL_OUTPUT_PATH = "Output_video/final_music_video.mp4"   # Video thành phẩm

# Kích thước video thành phẩm (Chuẩn YouTube Full HD)
OUT_WIDTH = 1920
OUT_HEIGHT = 1080
OUT_FPS = 60

# Vị trí đặt cụm sóng nhạc + ảnh nhân vật trên nền 1920x1080:
# - "(W-w)/2" : Căn chính giữa theo chiều ngang (hoặc đổi thành "80" nếu muốn đặt lệch trái để chừa chỗ làm lyric bên phải)
# - "(H-h)/2" : Căn chính giữa theo chiều dọc
OVERLAY_X = "(W-w)/2"
OVERLAY_Y = "(H-h)/2"

# Làm tối nền nhẹ (-0.15 đến 0.0) để sóng nhạc Neon nổi bật hơn
BG_BRIGHTNESS = -0.06
# =====================================================================


def compose_final_video():
    # Kiểm tra các file đầu vào
    for path in [SPECTRUM_MOV_PATH, BG_VIDEO_PATH, AUDIO_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {path}")

    os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)

    # Chuỗi filter FFmpeg:
    # 1. [0:v] Scale & crop video background về chuẩn 1920x1080, chỉnh FPS và độ sáng
    # 2. [1:v] Đặt đè lớp sóng nhạc + ảnh nhân vật (có sẵn kênh Alpha) lên trên nền
    filter_complex = (
        f"[0:v]scale={OUT_WIDTH}:{OUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUT_WIDTH}:{OUT_HEIGHT},fps={OUT_FPS},"
        f"eq=brightness={BG_BRIGHTNESS}[bg];"
        f"[bg][1:v]overlay={OVERLAY_X}:{OVERLAY_Y}:format=auto[outv]"
    )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",           # Tự động lặp lại video background nếu ngắn hơn bài nhạc
        "-i", BG_VIDEO_PATH,            # Input 0: Video background
        "-i", SPECTRUM_MOV_PATH,        # Input 1: Video sóng nhạc + ảnh nhân vật (.mov Alpha)
        "-i", AUDIO_PATH,               # Input 2: File nhạc gốc WAV
        "-filter_complex", filter_complex,
        "-map", "[outv]",               # Lấy hình ảnh đã ghép
        "-map", "2:a",                  # Lấy âm thanh trực tiếp từ file audio.WAV gốc
        "-c:v", "libx264",              # Dùng "h264_nvenc" nếu máy có GPU NVIDIA để xuất nhanh gấp 5 lần
        "-preset", "fast",
        "-crf", "18",                   # Chất lượng hình ảnh sắc nét chuẩn đăng YouTube
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",                 # Âm thanh AAC 320kbps chất lượng cao
        "-shortest",                    # Tự động dừng xuất khi hết thời lượng bài nhạc/sóng nhạc
        FINAL_OUTPUT_PATH
    ]

    print("Đang ghép sóng nhạc & ảnh nhân vật lên video background...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"\nHoàn tất! Video thành phẩm đã lưu tại: {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    compose_final_video()