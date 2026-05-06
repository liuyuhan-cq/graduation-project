import base64
import cgi
import csv
import http.server
import json
import os
import socketserver
import sys
import time
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

web_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(web_dir)
sys.path.insert(0, project_root)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

TORCH_IMPORT_ERROR = ""
LPIPS_IMPORT_ERROR = ""

try:
    import torch
except Exception as e:
    torch = None
    TORCH_IMPORT_ERROR = repr(e)

try:
    import lpips  # type: ignore
except Exception as e:
    lpips = None
    LPIPS_IMPORT_ERROR = repr(e)

MODEL_CACHE = {}
BATCH_RESULTS_CACHE = {}
LPIPS_METRIC = None
LPIPS_STATE = {"mode": "unknown", "reason": ""}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "weights")
BATCH_OUTPUT_DIR = os.path.join(BASE_DIR, "batch_outputs")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

MODEL_URLS = {
    "ESRGAN_x4": [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"
    ],
    "RealESRGAN_x4plus": [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ],
    "RealESRGAN_x4plus_anime_6B": [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    ],
    "realesr-general-x4v3": [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ],
}


class LPIPSMetric:
    def __init__(self):
        if lpips is None or torch is None:
            raise RuntimeError("lpips or torch unavailable")
        torch_cache_dir = os.path.join(WEIGHTS_DIR, "torch_cache")
        os.makedirs(torch_cache_dir, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", torch_cache_dir)
        self.device = torch.device("cpu")
        self.metric = lpips.LPIPS(net="alex").to(self.device).eval()

    @staticmethod
    def _to_tensor(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def calculate(self, sr_bgr, hr_bgr):
        # Avoid class-definition-time dependency on torch decorators.
        # This keeps server startup safe even when lpips/torch are unavailable.
        with torch.no_grad():
            sr_t = self._to_tensor(sr_bgr).to(self.device)
            hr_t = self._to_tensor(hr_bgr).to(self.device)
            return float(self.metric(sr_t, hr_t).item())


def get_lpips_metric():
    global LPIPS_METRIC
    if LPIPS_METRIC is not None:
        return LPIPS_METRIC

    if LPIPS_STATE["mode"] == "unavailable":
        return None

    if lpips is None or torch is None:
        LPIPS_STATE["mode"] = "unavailable"
        missing = []
        if torch is None:
            missing.append(f"torch_missing({TORCH_IMPORT_ERROR or 'import failed'})")
        if lpips is None:
            missing.append(f"lpips_missing({LPIPS_IMPORT_ERROR or 'import failed'})")
        LPIPS_STATE["reason"] = "; ".join(missing) if missing else "lpips_or_torch_missing"
        return None

    try:
        LPIPS_METRIC = LPIPSMetric()
        LPIPS_STATE["mode"] = "lpips"
    except Exception as e:
        LPIPS_STATE["mode"] = "unavailable"
        LPIPS_STATE["reason"] = str(e)
        LPIPS_METRIC = None
    return LPIPS_METRIC


def imencode_png_base64(img):
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buffer).decode("utf-8")


def b64_to_bytes(data):
    return base64.b64decode(data.encode("utf-8"))


def mod_crop(img, scale=4):
    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    return img[:h, :w, ...]


def make_lr_from_hr(hr, scale=4):
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def model_display_name(model_name):
    mapping = {
        "Bicubic": "Bicubic",
        "ESRGAN_x4": "ESRGAN_x4",
        "RealESRGAN_x4plus": "RealESRGAN_x4plus",
        "realesr-general-x4v3": "realesr-general-x4v3",
    }
    return mapping.get(model_name, model_name)


def make_compare_image(input_img, output_img, input_size, output_size):
    h1, w1 = input_img.shape[:2]
    h2, w2 = output_img.shape[:2]

    title_h = 95
    footer_h = 50
    pad = 20

    panel_h = max(h1, h2)
    panel_w = max(w1, w2)

    canvas_w = pad * 3 + panel_w * 2
    canvas_h = pad * 2 + title_h + panel_h + footer_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    cv2.putText(canvas, "Real-ESRGAN Compare", (pad, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Input", (pad, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Output", (pad * 2 + panel_w, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    def paste_center(dst, src, x0, y0, box_w, box_h):
        hh, ww = src.shape[:2]
        scale = min(box_w / ww, box_h / hh)
        nw, nh = int(ww * scale), int(hh * scale)
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        x = x0 + (box_w - nw) // 2
        y = y0 + (box_h - nh) // 2
        dst[y : y + nh, x : x + nw] = resized

    y0 = pad + title_h
    paste_center(canvas, input_img, pad, y0, panel_w, panel_h)
    paste_center(canvas, output_img, pad * 2 + panel_w, y0, panel_w, panel_h)

    in_text = f"Input: {input_size[0]} x {input_size[1]}"
    out_text = f"Output: {output_size[0]} x {output_size[1]}"
    cv2.putText(canvas, in_text, (pad, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, out_text, (pad * 2 + panel_w, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def make_test_compare_image(images_by_model):
    order = ["Preprocessed_LR", "Bicubic", "ESRGAN_x4", "RealESRGAN_x4plus", "realesr-general-x4v3"]
    imgs = [images_by_model[name] for name in order]
    panel_h = max(img.shape[0] for img in imgs)
    panel_w = max(img.shape[1] for img in imgs)

    pad = 16
    title_h = 78
    footer_h = 32
    canvas_w = pad * (len(order) + 1) + panel_w * len(order)
    canvas_h = pad * 2 + title_h + panel_h + footer_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    cv2.putText(canvas, "Five-View Horizontal Test Compare", (pad, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)

    for idx, name in enumerate(order):
        x0 = pad + idx * (panel_w + pad)
        y0 = pad + title_h
        img = images_by_model[name]
        ih, iw = img.shape[:2]
        scale = min(panel_w / iw, panel_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        dx = x0 + (panel_w - nw) // 2
        dy = y0 + (panel_h - nh) // 2
        canvas[dy : dy + nh, dx : dx + nw] = resized
        if name == "Preprocessed_LR":
            label = "Preprocessed LR(x4)"
        else:
            label = model_display_name(name)
        # Keep label inside each panel even for narrow widths.
        fs = max(0.35, min(0.72, panel_w / 250.0))
        thickness = 1 if fs < 0.5 else 2
        cv2.putText(canvas, label, (x0 + 4, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness, cv2.LINE_AA)

    return canvas


def download_weight(url, file_name):
    model_path = os.path.join(WEIGHTS_DIR, file_name)
    if os.path.isfile(model_path):
        return model_path
    return load_file_from_url(url=url, model_dir=WEIGHTS_DIR, progress=True, file_name=file_name)


def init_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if model_name == "Bicubic":
        MODEL_CACHE[model_name] = {"type": "bicubic", "scale": 4}
        return MODEL_CACHE[model_name]

    if model_name in ("RealESRGAN_x4plus", "ESRGAN_x4"):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        netscale = 4
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    urls = MODEL_URLS.get(model_name, [])
    if not urls:
        raise RuntimeError(f"No model URL configured for {model_name}")

    if model_name == "realesr-general-x4v3":
        model_paths = [download_weight(url, os.path.basename(url.split("?")[0])) for url in urls]
        model_path = model_paths
        dni_weight = [0.5, 0.5]
    else:
        model_path = download_weight(urls[0], os.path.basename(urls[0].split("?")[0]))
        dni_weight = None

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None,
    )
    MODEL_CACHE[model_name] = upsampler
    return upsampler


def process_image(img, model_name="RealESRGAN_x4plus"):
    if model_name == "Bicubic":
        h, w = img.shape[:2]
        return cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    upsampler = init_model(model_name)
    output, _ = upsampler.enhance(img, outscale=4)
    return output


def calc_metrics(sr, hr):
    h, w = hr.shape[:2]
    crop_border = 4 if min(h, w) > 12 else 0
    psnr = float(calculate_psnr(sr, hr, crop_border=crop_border, input_order="HWC", test_y_channel=False))
    ssim = float(calculate_ssim(sr, hr, crop_border=crop_border, input_order="HWC", test_y_channel=False))
    lpips_metric = get_lpips_metric()
    if lpips_metric is None:
        reason = LPIPS_STATE.get("reason", "lpips_not_ready")
        raise RuntimeError(
            f"LPIPS unavailable: {reason}. "
            f"Python={sys.executable}. 请在该解释器下安装并验证 torch 与 lpips。"
        )
    lpips_score = float(lpips_metric.calculate(sr, hr))
    return psnr, ssim, lpips_score


def metrics_to_csv(metrics):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["model", "psnr", "ssim", "lpips", "time_ms"])
    for item in metrics:
        writer.writerow(
            [
                item["model"],
                f'{item["psnr"]:.4f}' if item["psnr"] is not None else "",
                f'{item["ssim"]:.4f}' if item["ssim"] is not None else "",
                f'{item["lpips"]:.4f}' if item["lpips"] is not None else "",
                f'{item["time_ms"]:.4f}' if item["time_ms"] is not None else "",
            ]
        )
    return output.getvalue()


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Force-disable browser cache so updated frontend pages are always loaded.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/process":
            self.handle_single_process()
            return
        if parsed.path == "/batch_process":
            self.handle_batch_process()
            return
        if parsed.path == "/test_compare":
            self.handle_test_compare()
            return
        super().do_POST()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/batch_download_zip/"):
            self.handle_batch_zip_download(parsed.path)
            return
        if parsed.path.startswith("/batch_download/"):
            self.handle_batch_download(parsed.path)
            return
        super().do_GET()

    def handle_single_process(self):
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            if "file" not in form:
                self.send_json({"error": "未找到上传文件"})
                return

            file_item = form["file"]
            model_name = form.getvalue("model", "RealESRGAN_x4plus")
            img_data = file_item.file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                self.send_json({"error": "图像解码失败，请检查文件格式"})
                return

            input_h, input_w = img.shape[:2]
            output = process_image(img, model_name=model_name)
            if output is None:
                self.send_json({"error": "图像处理失败"})
                return

            output_h, output_w = output.shape[:2]
            output_base64 = imencode_png_base64(output)
            compare_img = make_compare_image(img, output, (input_w, input_h), (output_w, output_h))
            compare_base64 = imencode_png_base64(compare_img)

            self.send_json(
                {
                    "output": output_base64,
                    "compare": compare_base64,
                    "input_size": [input_w, input_h],
                    "output_size": [output_w, output_h],
                    "model": model_name,
                }
            )
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_batch_process(self):
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            model_name = form.getvalue("model", "RealESRGAN_x4plus")
            if "files" not in form:
                self.send_json({"error": "未找到批量上传字段 files"})
                return

            file_field = form["files"]
            files = file_field if isinstance(file_field, list) else [file_field]

            results = []
            batch_id = str(int(time.time()))
            batch_dir = os.path.join(BATCH_OUTPUT_DIR, batch_id)
            os.makedirs(batch_dir, exist_ok=True)

            for idx, file_item in enumerate(files):
                filename = os.path.basename(file_item.filename or f"file_{idx}.png")
                img_data = file_item.file.read()
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    results.append({"filename": filename, "error": "图像解码失败"})
                    continue

                input_h, input_w = img.shape[:2]
                output = process_image(img, model_name=model_name)
                if output is None:
                    results.append({"filename": filename, "error": "图像处理失败"})
                    continue

                output_h, output_w = output.shape[:2]
                stem = Path(filename).stem
                output_name = f"{stem}_sr.png"
                compare_name = f"{stem}_compare.png"
                output_path = os.path.join(batch_dir, output_name)
                compare_path = os.path.join(batch_dir, compare_name)

                compare_img = make_compare_image(img, output, (input_w, input_h), (output_w, output_h))
                cv2.imwrite(output_path, output)
                cv2.imwrite(compare_path, compare_img)

                results.append(
                    {
                        "filename": filename,
                        "output_file": output_name,
                        "compare_file": compare_name,
                        "input_size": [input_w, input_h],
                        "output_size": [output_w, output_h],
                        "model": model_name,
                        "output_base64": imencode_png_base64(output),
                        "compare_base64": imencode_png_base64(compare_img),
                    }
                )

            BATCH_RESULTS_CACHE[batch_id] = {"created_at": time.time(), "results": results}
            self.send_json({"batch_id": batch_id, "model": model_name, "count": len(results), "results": results})
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_test_compare(self):
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            if "file" not in form:
                self.send_json({"error": "未找到测试图片"})
                return

            file_item = form["file"]
            img_data = file_item.file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            hr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if hr is None:
                self.send_json({"error": "图像解码失败"})
                return

            hr = mod_crop(hr, 4)
            h, w = hr.shape[:2]
            if h < 16 or w < 16:
                self.send_json({"error": "图片尺寸过小，请上传更大图片"})
                return

            lr = make_lr_from_hr(hr, 4)
            lr_h, lr_w = lr.shape[:2]
            lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)

            test_models = ["Bicubic", "ESRGAN_x4", "RealESRGAN_x4plus", "realesr-general-x4v3"]
            metrics = []
            sr_images = {"Preprocessed_LR": lr_up}

            for model_name in test_models:
                t0 = time.perf_counter()
                sr = process_image(lr, model_name=model_name)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if sr.shape[0] != h or sr.shape[1] != w:
                    sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_CUBIC)

                psnr, ssim, lpips_score = calc_metrics(sr, hr)
                metrics.append(
                    {
                        "model": model_display_name(model_name),
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lpips_score,
                        "time_ms": elapsed_ms,
                    }
                )
                sr_images[model_name] = sr

            compare_img = make_test_compare_image(sr_images)
            compare_base64 = imencode_png_base64(compare_img)
            csv_text = metrics_to_csv(metrics)

            self.send_json(
                {
                    "compare": compare_base64,
                    "metrics": metrics,
                    "csv_text": csv_text,
                    "input_size": [w, h],
                    "lr_size": [lr_w, lr_h],
                    "lpips_mode": LPIPS_STATE.get("mode", "unknown"),
                }
            )
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_batch_download(self, path):
        try:
            parts = path.strip("/").split("/")
            if len(parts) < 3:
                self.send_error(404, "Invalid download path")
                return

            _, batch_id, filename = parts[0], parts[1], "/".join(parts[2:])
            file_path = os.path.join(BATCH_OUTPUT_DIR, batch_id, filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "File not found")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(os.path.getsize(file_path)))
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, str(e))

    def handle_batch_zip_download(self, path):
        try:
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self.send_error(404, "Invalid zip download path")
                return

            _, batch_id, file_type = parts
            safe_batch_id = os.path.basename(batch_id)

            if file_type == "output":
                zip_name = f"batch_{safe_batch_id}_outputs.zip"
                b64_key = "output_base64"
                suffix = "_sr.png"
            elif file_type == "compare":
                zip_name = f"batch_{safe_batch_id}_compares.zip"
                b64_key = "compare_base64"
                suffix = "_compare.png"
            else:
                self.send_error(400, "Invalid zip file type")
                return

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                written = 0
                cache_entry = BATCH_RESULTS_CACHE.get(safe_batch_id)
                if cache_entry:
                    for item in cache_entry.get("results", []):
                        if item.get("error"):
                            continue
                        b64_data = item.get(b64_key)
                        if not b64_data:
                            continue
                        stem = Path(item.get("filename", f"file_{written}.png")).stem
                        zf.writestr(f"{stem}{suffix}", b64_to_bytes(b64_data))
                        written += 1

                if written == 0:
                    batch_dir = os.path.join(BATCH_OUTPUT_DIR, safe_batch_id)
                    if not os.path.isdir(batch_dir):
                        self.send_error(404, "Batch not found")
                        return
                    files = sorted([f for f in os.listdir(batch_dir) if f.lower().endswith(suffix)])
                    for name in files:
                        zf.write(os.path.join(batch_dir, name), arcname=name)
                        written += 1

                if written == 0:
                    self.send_error(404, "No files for requested zip type")
                    return

            zip_bytes = zip_buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{zip_name}"')
            self.send_header("Content-Length", str(len(zip_bytes)))
            self.end_headers()
            self.wfile.write(zip_bytes)
        except Exception as e:
            self.send_error(500, str(e))

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def run_server():
    port = 8000
    os.chdir(BASE_DIR)
    with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
        print(f"Server running at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
