import os
import subprocess
import numpy as np
import librosa
from scipy.interpolate import make_interp_spline
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm

# ================= CẤU HÌNH ĐỒ HỌA & ẢNH NHÂN VẬT =================
AUDIO_PATH = "Input_audio/audio.WAV"            # File nhạc (.wav, .mp3, .flac...)
AVATAR_PATH = "Input_image/image.jpg"           # Ảnh nhân vật đặt ở chính giữa

# Chế độ xuất:
# - "mp4_black": Xuất file .mp4 nền đen
# - "mov_alpha": Xuất file .mov nền trong suốt 100% chuẩn QuickTime PNG (nhẹ, nhanh, giữ nguyên độ nét)
EXPORT_MODE = "mov_alpha"
OUTPUT_PATH = "Output_audio/acv_card_spectrum.mp4" if EXPORT_MODE == "mp4_black" else "Output_audio/acv_card_spectrum.mov"

WIDTH, HEIGHT = 1280, 1080
FPS = 60
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# Kích thước khung ảnh trung tâm & Cột sóng
CARD_WIDTH = 500                    # Chiều rộng/cao khung ảnh vuông (500x500)
CARD_CORNER_RADIUS = 38             # Độ bo tròn 4 góc khung ảnh
COLUMN_HEIGHT = 460                 # Chiều cao dải sóng dọc theo 2 cạnh ảnh

# Thông số thanh sóng (Giữ nguyên bản Pro)
NUM_BARS_PER_SIDE = 44
BAR_THICKNESS = 6
MIN_BAR_LEN = 12
MAX_BAR_LEN = 225

# Bảng màu Neon Rực (Lõi trắng + Đỏ hồng rực rỡ)
BAR_CORE_COLOR = (255, 255, 255, 255)
PEAK_CAP_COLOR = (255, 220, 230, 255)
WAVE_BACK_COLOR = (255, 30, 86, 170)
GLOW_INNER_COLOR = (255, 50, 100, 240)
CARD_BORDER_COLOR = (255, 190, 210, 235)   # Màu viền khung ảnh
CARD_GLOW_COLOR = (255, 35, 90, 200)       # Quầng sáng tỏa ra sau khung ảnh
# ==================================================================


def prepare_center_card(image_path, size, radius):
    """Cắt ảnh nhân vật về hình vuông, bo góc tròn và tạo viền phát sáng Neon."""
    img = Image.open(image_path).convert("RGBA")
    img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, size, size), radius=radius, fill=255)
    img.putalpha(mask)

    pad = 40
    canvas_size = size + pad * 2
    card_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    glow_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.rounded_rectangle(
        (pad - 3, pad - 3, pad + size + 3, pad + size + 3),
        radius=radius + 3,
        outline=CARD_GLOW_COLOR,
        width=8
    )
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=14))

    card_canvas = Image.alpha_composite(card_canvas, glow_layer)
    card_canvas.paste(img, (pad, pad), img)

    border_draw = ImageDraw.Draw(card_canvas)
    border_draw.rounded_rectangle(
        (pad, pad, pad + size, pad + size),
        radius=radius,
        outline=CARD_BORDER_COLOR,
        width=3
    )
    return card_canvas


def analyze_audio_pro(audio_path, fps, num_bars):
    """Phân tích phổ tần số và năng lượng Bass."""
    print("Đang phân tích tín hiệu âm thanh...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    total_frames = int(duration * fps)

    hop_length = int(sr / fps)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # 1. Năng lượng Bass (20Hz - 150Hz)
    bass_idx = np.where((freqs >= 20) & (freqs <= 150))[0]
    bass_env = np.mean(stft[bass_idx, :], axis=0)
    bass_env = np.clip(bass_env / (np.percentile(bass_env, 97) + 1e-6), 0, 1.3)
    bass_env = np.power(bass_env, 1.2)

    # 2. Chia dải tần đối xứng từ tâm ra 2 đầu
    half_bars = num_bars // 2
    freq_bins = np.geomspace(35, 5500, half_bars + 1)
    half_spec = np.zeros((half_bars, stft.shape[1]))

    for i in range(half_bars):
        idx = np.where((freqs >= freq_bins[i]) & (freqs < freq_bins[i + 1]))[0]
        if len(idx) > 0:
            half_spec[i] = np.mean(stft[idx, :], axis=0)

    max_vals = np.percentile(half_spec, 96, axis=1, keepdims=True) + 1e-6
    half_spec = np.clip(half_spec / max_vals, 0, 1.25)
    half_spec = np.power(half_spec, 0.65)

    full_col_spec = np.vstack([half_spec[::-1], half_spec])
    kernel = np.array([0.2, 0.6, 0.2])
    for t in range(full_col_spec.shape[1]):
        full_col_spec[:, t] = np.convolve(full_col_spec[:, t], kernel, mode="same")

    arch_window = np.hanning(num_bars) * 0.75 + 0.25
    full_col_spec = full_col_spec * arch_window[:, np.newaxis]

    actual_frames = full_col_spec.shape[1]
    if actual_frames < total_frames:
        pad = total_frames - actual_frames
        bass_env = np.pad(bass_env, (0, pad))
        full_col_spec = np.pad(full_col_spec, ((0, 0), (0, pad)))
    else:
        bass_env = bass_env[:total_frames]
        full_col_spec = full_col_spec[:, :total_frames]

    for t in range(1, total_frames):
        alpha_b = 0.85 if bass_env[t] > bass_env[t - 1] else 0.45
        bass_env[t] = alpha_b * bass_env[t] + (1 - alpha_b) * bass_env[t - 1]

        for b in range(num_bars):
            alpha_s = 0.82 if full_col_spec[b, t] > full_col_spec[b, t - 1] else 0.48
            full_col_spec[b, t] = alpha_s * full_col_spec[b, t] + (1 - alpha_s) * full_col_spec[b, t - 1]

    return bass_env, full_col_spec, total_frames


class ParticleSystem:
    """Quản lý hạt bụi ánh sáng và tam giác hình học bay tỏa ra từ 2 bên sóng nhạc."""
    def __init__(self, num_particles=80, num_triangles=12):
        rng = np.random.default_rng(42)
        self.num_p = num_particles
        self.particles = np.zeros((num_particles, 8))
        for i in range(num_particles):
            self._respawn_particle(i, rng, random_progress=True)
        self.rng = rng

        self.triangles = []
        for i in range(num_triangles):
            side = -1 if i % 2 == 0 else 1
            x = CENTER_X + side * rng.uniform(CARD_WIDTH // 2 + 50, CARD_WIDTH // 2 + 320)
            y = CENTER_Y + rng.uniform(-260, 260)
            vx = side * rng.uniform(0.3, 0.9)
            vy = rng.uniform(-0.5, 0.5)
            radius = rng.uniform(18, 46)
            angle = rng.uniform(0, 2 * np.pi)
            v_angle = rng.uniform(-0.025, 0.025)
            self.triangles.append([x, y, vx, vy, radius, angle, v_angle, side])

    def _respawn_particle(self, idx, rng, random_progress=False):
        side = -1 if idx % 2 == 0 else 1
        anchor_x = CENTER_X + side * (CARD_WIDTH // 2 + 6)
        y = CENTER_Y + rng.uniform(-COLUMN_HEIGHT // 2, COLUMN_HEIGHT // 2)
        vx = side * rng.uniform(1.2, 4.2)
        vy = rng.uniform(-0.8, 0.8)
        size = rng.uniform(1.5, 4.0)
        max_life = rng.uniform(45, 110)
        life = rng.uniform(0, max_life) if random_progress else 0
        x = anchor_x + vx * life
        self.particles[idx] = [x, y, vx, vy, size, life, max_life, side]

    def update_and_draw(self, draw_core, draw_glow, bass_val):
        speed_boost = 1.0 + bass_val * 2.4

        for i in range(self.num_p):
            p = self.particles[i]
            p[0] += p[2] * speed_boost
            p[1] += p[3]
            p[5] += 1
            if p[5] >= p[6] or p[0] < 20 or p[0] > WIDTH - 20:
                self._respawn_particle(i, self.rng)
                p = self.particles[i]

            alpha_ratio = np.sin((p[5] / p[6]) * np.pi)
            alpha = int(230 * alpha_ratio)
            r = p[4] * (1.0 + bass_val * 0.35)
            bbox = [p[0] - r, p[1] - r, p[0] + r, p[1] + r]
            draw_core.ellipse(bbox, fill=(255, 230, 240, alpha))
            draw_glow.ellipse([p[0] - r * 2, p[1] - r * 2, p[0] + r * 2, p[1] + r * 2],
                              fill=(255, 40, 95, int(alpha * 0.8)))

        centers = []
        for tri in self.triangles:
            tri[0] += tri[2] * (1.0 + bass_val * 1.2)
            tri[1] += tri[3]
            tri[5] += tri[6] * (1.0 + bass_val)

            min_x = CENTER_X + tri[7] * (CARD_WIDTH // 2 + 40)
            max_x = CENTER_X + tri[7] * (CARD_WIDTH // 2 + 340)
            if (tri[7] == -1 and tri[0] < max_x) or (tri[7] == 1 and tri[0] > max_x):
                tri[0] = min_x
                tri[1] = CENTER_Y + self.rng.uniform(-250, 250)
            if abs(tri[1] - CENTER_Y) > 280:
                tri[3] *= -1

            r = tri[4] * (1.0 + bass_val * 0.25)
            pts = [
                (tri[0] + r * np.cos(tri[5] + k * 2 * np.pi / 3),
                 tri[1] + r * np.sin(tri[5] + k * 2 * np.pi / 3))
                for k in range(3)
            ]
            draw_core.polygon(pts, outline=(255, 220, 230, 150), width=2)
            draw_glow.polygon(pts, outline=(255, 40, 95, 180), width=3)
            centers.append((tri[0], tri[1], tri[7]))

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if centers[i][2] == centers[j][2]:
                    dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                    if dist < 170:
                        line_alpha = int(110 * (1 - dist / 170))
                        draw_glow.line([centers[i][:2], centers[j][:2]],
                                       fill=(255, 60, 110, line_alpha), width=1)


def render_frame_with_avatar(bars, peak_caps, bass_val, particles_sys, card_base):
    """Vẽ sóng nhạc Pro và ghép khớp khung ảnh nhân vật co giãn theo nhịp Bass ở trung tâm."""
    core_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    glow_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    wave_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))

    draw_core = ImageDraw.Draw(core_layer)
    draw_glow = ImageDraw.Draw(glow_layer)
    draw_wave = ImageDraw.Draw(wave_layer)

    scale = 1.0 + (bass_val * 0.045)
    current_half_card = int((CARD_WIDTH * scale) / 2)
    current_col_height = COLUMN_HEIGHT * scale

    left_anchor_x = CENTER_X - current_half_card
    right_anchor_x = CENTER_X + current_half_card

    start_y = CENTER_Y - (current_col_height / 2)
    y_coords = np.linspace(start_y, start_y + current_col_height, NUM_BARS_PER_SIDE)
    bar_lengths = MIN_BAR_LEN + (bars * MAX_BAR_LEN)

    # --- LỚP 1: DẢI SÓNG LỤA LIỀN MẠCH PHÍA SAU ---
    y_smooth = np.linspace(start_y, start_y + current_col_height, 200)
    spline = make_interp_spline(y_coords, bar_lengths * 1.15, k=3)
    len_smooth = np.maximum(spline(y_smooth), MIN_BAR_LEN)

    left_poly = [(left_anchor_x, start_y)] + \
                [(left_anchor_x - l, y) for y, l in zip(y_smooth, len_smooth)] + \
                [(left_anchor_x, start_y + current_col_height)]
    right_poly = [(right_anchor_x, start_y)] + \
                 [(right_anchor_x + l, y) for y, l in zip(y_smooth, len_smooth)] + \
                 [(right_anchor_x, start_y + current_col_height)]

    draw_wave.polygon(left_poly, fill=WAVE_BACK_COLOR)
    draw_wave.polygon(right_poly, fill=WAVE_BACK_COLOR)
    wave_layer = wave_layer.filter(ImageFilter.GaussianBlur(radius=12))

    # --- LỚP 2: CÁC THANH BAR CHÍNH + CHẤM ĐỈNH RƠI ---
    for i in range(NUM_BARS_PER_SIDE):
        y = y_coords[i]
        b_len = bar_lengths[i]
        p_len = MIN_BAR_LEN + (peak_caps[i] * MAX_BAR_LEN) + 8

        # Bên trái
        lx1, lx2 = left_anchor_x - b_len, left_anchor_x + 6
        ly1, ly2 = y - BAR_THICKNESS / 2, y + BAR_THICKNESS / 2
        draw_core.rounded_rectangle([lx1, ly1, lx2, ly2], radius=3, fill=BAR_CORE_COLOR)
        draw_glow.rounded_rectangle([lx1 - 4, ly1 - 3, lx2, ly2 + 3], radius=5, fill=GLOW_INNER_COLOR)
        px_l = left_anchor_x - p_len
        draw_core.rounded_rectangle([px_l - 5, ly1, px_l, ly2], radius=2, fill=PEAK_CAP_COLOR)

        # Bên phải
        rx1, rx2 = right_anchor_x - 6, right_anchor_x + b_len
        ry1, ry2 = y - BAR_THICKNESS / 2, y + BAR_THICKNESS / 2
        draw_core.rounded_rectangle([rx1, ry1, rx2, ry2], radius=3, fill=BAR_CORE_COLOR)
        draw_glow.rounded_rectangle([rx1, ry1 - 3, rx2 + 4, ry2 + 3], radius=5, fill=GLOW_INNER_COLOR)
        px_r = right_anchor_x + p_len
        draw_core.rounded_rectangle([px_r, ry1, px_r + 5, ry2], radius=2, fill=PEAK_CAP_COLOR)

    # --- LỚP 3: HẠT ÁNH SÁNG & TAM GIÁC PLEXUS ---
    particles_sys.update_and_draw(draw_core, draw_glow, bass_val)

    # --- GỘP CÁC LỚP SÁNG NEON ---
    wide_bloom = glow_layer.filter(ImageFilter.GaussianBlur(radius=24))
    tight_bloom = glow_layer.filter(ImageFilter.GaussianBlur(radius=7))
    core_soft = core_layer.filter(ImageFilter.GaussianBlur(radius=2))

    result = Image.alpha_composite(wave_layer, wide_bloom)
    result = Image.alpha_composite(result, tight_bloom)
    result = Image.alpha_composite(result, core_soft)
    result = Image.alpha_composite(result, core_layer)

    # --- LỚP TRÊN CÙNG: GHÉP KHUNG ẢNH NHÂN VẬT CHUẨN ALPHA ---
    new_card_w = int(card_base.width * scale)
    card_scaled = card_base.resize((new_card_w, new_card_w), Image.Resampling.BILINEAR)

    card_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    paste_x = CENTER_X - (new_card_w // 2)
    paste_y = CENTER_Y - (new_card_w // 2)
    card_layer.paste(card_scaled, (paste_x, paste_y))

    result = Image.alpha_composite(result, card_layer)

    if EXPORT_MODE == "mp4_black":
        black_bg = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
        result = Image.alpha_composite(black_bg, result)

    return result


def main():
    # Tự động tạo thư mục chứa file output nếu chưa tồn tại
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    bass_env, full_col_spec, total_frames = analyze_audio_pro(AUDIO_PATH, FPS, NUM_BARS_PER_SIDE)
    card_base = prepare_center_card(AVATAR_PATH, CARD_WIDTH, CARD_CORNER_RADIUS)
    particles_sys = ParticleSystem(num_particles=80, num_triangles=12)
    peak_caps = np.zeros(NUM_BARS_PER_SIDE)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgba",
        "-r", str(FPS),
        "-i", "-",
        "-i", AUDIO_PATH,
    ]

    if EXPORT_MODE == "mov_alpha":
        # Cách 1: QuickTime PNG (.mov) - giữ nguyên tốc độ & chất lượng Lossless, giảm 97% dung lượng
        ffmpeg_cmd.extend([
            "-c:v", "png",
            "-pix_fmt", "rgba",
            "-c:a", "aac", "-b:a", "320k",
            "-shortest", OUTPUT_PATH
        ])
    else:
        ffmpeg_cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "320k",
            "-shortest", OUTPUT_PATH
        ])

    print(f"Đang render cụm sóng nhạc + ảnh nhân vật ({EXPORT_MODE}) -> {OUTPUT_PATH}...")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for f in tqdm(range(total_frames), unit="frame"):
        bars = full_col_spec[:, f]
        peak_caps = np.maximum(bars, peak_caps - 0.018)

        frame_img = render_frame_with_avatar(bars, peak_caps, bass_env[f], particles_sys, card_base)
        process.stdin.write(frame_img.tobytes())

    process.stdin.close()
    process.wait()
    print(f"\nHoàn tất! File đã được lưu tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    main() 
    from compose_video import compose_final_video
    compose_final_video()