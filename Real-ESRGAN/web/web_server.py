# web_server.py
import http.server
import socketserver
import os
import sys
import cgi
import cv2
import numpy as np
import base64
import json
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
import zipfile
import time

# 添加项目根目录到 Python 路径
web_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(web_dir)
sys.path.insert(0, project_root)
print(f"Added project root to sys.path: {project_root}")
print(f"Current sys.path: {sys.path[:5]}")

# 尝试导入必要的模块
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    print("Successfully imported basicsr modules")
except ImportError as e:
    print(f"Failed to import basicsr: {e}")

try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    print("Successfully imported realesrgan modules")
except ImportError as e:
    print(f"Failed to import realesrgan: {e}")

# 全局模型缓存，避免每次请求重复加载
MODEL_CACHE = {}
BATCH_RESULTS_CACHE = {}

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, '..', 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 批量处理输出目录
BATCH_OUTPUT_DIR = os.path.join(BASE_DIR, 'batch_outputs')
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)


def init_model(model_name):
    """初始化 Real-ESRGAN 模型，带缓存机制"""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        default_pth = 'RealESRGAN_x4plus.pth'

    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        default_pth = 'RealESRGAN_x4plus_anime_6B.pth'

    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
        default_pth = 'realesr-general-x4v3.pth'

    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        default_pth = 'RealESRGAN_x4plus.pth'

    model_path = os.path.join(WEIGHTS_DIR, default_pth)

    # 如果本地没有权重，就下载
    if not os.path.isfile(model_path):
        for url in file_url:
            try:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=WEIGHTS_DIR,
                    progress=True,
                    file_name=default_pth
                )
                break
            except Exception as e:
                print(f"Download failed from {url}: {e}")

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None
    )

    MODEL_CACHE[model_name] = upsampler
    return upsampler


def process_image(img, model_name='RealESRGAN_x4plus'):
    """图像超分处理，仅使用 Real-ESRGAN"""
    upsampler = init_model(model_name)
    output, _ = upsampler.enhance(img, outscale=4)
    return output


def imencode_png_base64(img):
    ok, buffer = cv2.imencode('.png', img)
    if not ok:
        raise RuntimeError('图像编码失败')
    return base64.b64encode(buffer).decode('utf-8')


def b64_to_bytes(data):
    return base64.b64decode(data.encode('utf-8'))


def make_compare_image(input_img, output_img, input_size, output_size):
    """生成原图+结果图的拼接对比图"""
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

    # 标题
    cv2.putText(canvas, 'Real-ESRGAN Compare', (pad, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # 子标题
    cv2.putText(canvas, 'Input', (pad, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Output', (pad * 2 + panel_w, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    def paste_center(dst, src, x0, y0, box_w, box_h):
        hh, ww = src.shape[:2]
        scale = min(box_w / ww, box_h / hh)
        nw, nh = int(ww * scale), int(hh * scale)
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        x = x0 + (box_w - nw) // 2
        y = y0 + (box_h - nh) // 2
        dst[y:y + nh, x:x + nw] = resized

    y0 = pad + title_h
    paste_center(canvas, input_img, pad, y0, panel_w, panel_h)
    paste_center(canvas, output_img, pad * 2 + panel_w, y0, panel_w, panel_h)

    # 底部信息
    in_text = f'Input: {input_size[0]} x {input_size[1]}'
    out_text = f'Output: {output_size[0]} x {output_size[1]}'
    cv2.putText(canvas, in_text, (pad, canvas_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, out_text, (pad * 2 + panel_w, canvas_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/process':
            self.handle_single_process()
            return

        if parsed.path == '/batch_process':
            self.handle_batch_process()
            return

        super().do_POST()

    def handle_single_process(self):
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )

            if 'file' not in form:
                self.send_json({'error': '未找到上传文件'})
                return

            file_item = form['file']
            model_name = form.getvalue('model', 'RealESRGAN_x4plus')

            img_data = file_item.file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                self.send_json({'error': '图像解码失败，请检查上传文件格式'})
                return

            input_h, input_w = img.shape[:2]

            output = process_image(img, model_name=model_name)

            if output is None:
                self.send_json({'error': '图像处理失败，输出为空'})
                return

            output_h, output_w = output.shape[:2]
            print(f'Input size: {input_w}x{input_h}, Output size: {output_w}x{output_h}')

            output_base64 = imencode_png_base64(output)

            compare_img = make_compare_image(img, output, (input_w, input_h), (output_w, output_h))
            compare_base64 = imencode_png_base64(compare_img)

            self.send_json({
                'output': output_base64,
                'compare': compare_base64,
                'input_size': [input_w, input_h],
                'output_size': [output_w, output_h],
                'model': model_name
            })

        except Exception as e:
            self.send_json({'error': str(e)}, status=500)

    def handle_batch_process(self):
        """
        批量处理接口：
        前端通过 multipart/form-data 发送多个文件，字段名为 files
        可选字段：model
        返回每张图的结果与对比图 base64
        """
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )

            model_name = form.getvalue('model', 'RealESRGAN_x4plus')

            if 'files' not in form:
                self.send_json({'error': '未找到批量上传文件字段 files'})
                return

            file_field = form['files']

            # 兼容单个文件与多个文件
            files = file_field if isinstance(file_field, list) else [file_field]

            results = []
            batch_id = str(int(time.time()))
            batch_dir = os.path.join(BATCH_OUTPUT_DIR, batch_id)
            os.makedirs(batch_dir, exist_ok=True)

            for idx, file_item in enumerate(files):
                filename = os.path.basename(file_item.filename or f'file_{idx}.png')
                img_data = file_item.file.read()
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    results.append({
                        'filename': filename,
                        'error': '图像解码失败'
                    })
                    continue

                input_h, input_w = img.shape[:2]

                output = process_image(img, model_name=model_name)
                if output is None:
                    results.append({
                        'filename': filename,
                        'error': '图像处理失败'
                    })
                    continue

                output_h, output_w = output.shape[:2]

                stem = Path(filename).stem
                output_name = f'{stem}_sr.png'
                compare_name = f'{stem}_compare.png'

                output_path = os.path.join(batch_dir, output_name)
                compare_path = os.path.join(batch_dir, compare_name)

                cv2.imwrite(output_path, output)
                compare_img = make_compare_image(img, output, (input_w, input_h), (output_w, output_h))
                cv2.imwrite(compare_path, compare_img)

                results.append({
                    'filename': filename,
                    'output_file': output_name,
                    'compare_file': compare_name,
                    'input_size': [input_w, input_h],
                    'output_size': [output_w, output_h],
                    'model': model_name,
                    'output_base64': imencode_png_base64(output),
                    'compare_base64': imencode_png_base64(compare_img)
                })

            self.send_json({
                'batch_id': batch_id,
                'model': model_name,
                'count': len(results),
                'results': results
            })

            # 缓存批量结果，用于 ZIP 下载时直接从内存打包，避免中文路径落盘失败导致文件缺失
            BATCH_RESULTS_CACHE[batch_id] = {
                'created_at': time.time(),
                'results': results
            }

        except Exception as e:
            self.send_json({'error': str(e)}, status=500)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path.startswith('/batch_download_zip/'):
            self.handle_batch_zip_download(parsed.path)
            return

        if parsed.path.startswith('/batch_download/'):
            self.handle_batch_download(parsed.path)
            return

        super().do_GET()

    def handle_batch_download(self, path):
        """
        下载批量处理结果文件
        URL 示例：
        /batch_download/<batch_id>/<filename>
        """
        try:
            parts = path.strip('/').split('/')
            if len(parts) < 3:
                self.send_error(404, 'Invalid download path')
                return

            _, batch_id, filename = parts[0], parts[1], '/'.join(parts[2:])
            file_path = os.path.join(BATCH_OUTPUT_DIR, batch_id, filename)

            if not os.path.isfile(file_path):
                self.send_error(404, 'File not found')
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            fs = os.path.getsize(file_path)
            self.send_header('Content-Length', str(fs))
            self.end_headers()

            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, str(e))

    def handle_batch_zip_download(self, path):
        """
        下载批量结果的 ZIP 包
        URL 示例：
        /batch_download_zip/<batch_id>/output
        /batch_download_zip/<batch_id>/compare
        """
        try:
            parts = path.strip('/').split('/')
            if len(parts) != 3:
                self.send_error(404, 'Invalid zip download path')
                return

            _, batch_id, file_type = parts
            safe_batch_id = os.path.basename(batch_id)

            if file_type == 'output':
                zip_name = f'batch_{safe_batch_id}_outputs.zip'
                b64_key = 'output_base64'
                suffix = '_sr.png'
            elif file_type == 'compare':
                zip_name = f'batch_{safe_batch_id}_compares.zip'
                b64_key = 'compare_base64'
                suffix = '_compare.png'
            else:
                self.send_error(400, 'Invalid zip file type')
                return

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                written = 0

                # 优先从内存缓存写入 ZIP（更稳定）
                cache_entry = BATCH_RESULTS_CACHE.get(safe_batch_id)
                if cache_entry:
                    for item in cache_entry.get('results', []):
                        if item.get('error'):
                            continue
                        b64_data = item.get(b64_key)
                        if not b64_data:
                            continue
                        stem = Path(item.get('filename', f'file_{written}.png')).stem
                        name = f'{stem}{suffix}'
                        zf.writestr(name, b64_to_bytes(b64_data))
                        written += 1

                # 缓存没有命中时，回退到磁盘文件
                if written == 0:
                    batch_dir = os.path.join(BATCH_OUTPUT_DIR, safe_batch_id)
                    if not os.path.isdir(batch_dir):
                        self.send_error(404, 'Batch not found')
                        return
                    files = sorted([f for f in os.listdir(batch_dir) if f.lower().endswith(suffix)])
                    for name in files:
                        file_path = os.path.join(batch_dir, name)
                        zf.write(file_path, arcname=name)
                        written += 1

                if written == 0:
                    self.send_error(404, 'No files for requested zip type')
                    return
            zip_bytes = zip_buffer.getvalue()

            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', f'attachment; filename="{zip_name}"')
            self.send_header('Content-Length', str(len(zip_bytes)))
            self.end_headers()
            self.wfile.write(zip_bytes)
        except Exception as e:
            self.send_error(500, str(e))

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


def run_server():
    PORT = 8000
    os.chdir(BASE_DIR)

    handler = MyHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"服务器运行在 http://localhost:{PORT}")
        print("按 Ctrl+C 停止服务器")
        httpd.serve_forever()


if __name__ == '__main__':
    run_server()
