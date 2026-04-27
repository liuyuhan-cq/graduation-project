import os
import csv
import cv2
import argparse
import numpy as np
import torch
from typing import List, Optional, Dict

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import lpips  # type: ignore
except ImportError:
    lpips = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

MODEL_CONFIG = {
    'Bicubic': {
        'arch': 'BICUBIC',
        'scale': 4,
        'urls': []
    },
    'ESRGAN_x4': {
        'arch': 'RRDBNet23',
        'scale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
        ]
    },
    'RealESRGAN_x4plus': {
        'arch': 'RRDBNet23',
        'scale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        ]
    },
    'RealESRGAN_x4plus_anime_6B': {
        'arch': 'RRDBNet6',
        'scale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        ]
    },
    'realesr-general-x4v3': {
        'arch': 'SRVGG32',
        'scale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    }
}


class BicubicUpsampler:
    def __init__(self, scale: int):
        self.scale = scale

    def enhance(self, img: np.ndarray, outscale: Optional[float] = None):
        scale = float(outscale) if outscale is not None else float(self.scale)
        h, w = img.shape[:2]
        out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
        return out, None


class LPIPSMetric:
    def __init__(self, use_cpu: bool = False):
        if lpips is None:
            raise ImportError('lpips is not installed. Please run: python -m pip install lpips')
        # Force torch/torchvision model cache into project-local writable path.
        torch_cache_dir = os.path.join(WEIGHTS_DIR, 'torch_cache')
        os.makedirs(torch_cache_dir, exist_ok=True)
        os.environ.setdefault('TORCH_HOME', torch_cache_dir)
        self.device = torch.device('cpu' if use_cpu or (not torch.cuda.is_available()) else 'cuda')
        self.metric = lpips.LPIPS(net='alex').to(self.device).eval()

    @staticmethod
    def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor

    @torch.no_grad()
    def calculate(self, sr_bgr: np.ndarray, hr_bgr: np.ndarray) -> float:
        sr_t = self._to_tensor(sr_bgr).to(self.device)
        hr_t = self._to_tensor(hr_bgr).to(self.device)
        return float(self.metric(sr_t, hr_t).item())


def build_network(model_name: str):
    if model_name in ('RealESRGAN_x4plus', 'ESRGAN_x4', 'RealESRNet_x4plus'):
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if model_name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4
    if model_name == 'realesr-general-x4v3':
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'), 4
    if model_name == 'Bicubic':
        return None, 4
    raise ValueError(f'Unsupported model: {model_name}')


def download_weights(urls: List[str]) -> List[str]:
    paths = []
    for url in urls:
        file_name = os.path.basename(url.split('?')[0])
        path = os.path.join(WEIGHTS_DIR, file_name)
        if not os.path.isfile(path):
            print(f'Downloading weight: {file_name}')
            try:
                path = load_file_from_url(url=url, model_dir=WEIGHTS_DIR, progress=True, file_name=file_name)
            except Exception as e:
                raise RuntimeError(
                    f'Failed to download {file_name}. '
                    f'Please manually place it under {WEIGHTS_DIR}. Original error: {e}'
                ) from e
        paths.append(path)
    return paths


def init_model(model_name: str, use_cpu: bool = False, denoise_strength: float = 0.5):
    model, netscale = build_network(model_name)
    if model_name == 'Bicubic':
        return BicubicUpsampler(scale=netscale)

    weight_paths = download_weights(MODEL_CONFIG[model_name]['urls'])
    half = (not use_cpu) and torch.cuda.is_available()
    gpu_id = None if use_cpu or (not torch.cuda.is_available()) else 0

    if model_name == 'realesr-general-x4v3' and len(weight_paths) == 2:
        model_path = weight_paths
        dni_weight = [denoise_strength, 1 - denoise_strength]
    else:
        model_path = weight_paths[0]
        dni_weight = None

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half,
        gpu_id=gpu_id
    )
    return upsampler


def list_images(folder: str) -> List[str]:
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(exts):
            files.append(os.path.join(folder, fn))
    return files


def mod_crop(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    return img[:h, :w, ...]


def make_lr_from_hr(hr: np.ndarray, scale: int) -> np.ndarray:
    hr = mod_crop(hr, scale)
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def save_triplet_visual(lr: np.ndarray, sr: np.ndarray, hr: np.ndarray, save_path: str):
    h, w = hr.shape[:2]
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)
    canvas = np.concatenate([lr_up, sr, hr], axis=1)
    cv2.putText(canvas, 'LR(x4 upsampled)', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, 'SR', (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, 'HR(GT)', (2 * w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(save_path, canvas)


def get_default_lr_dir(hr_dir: str, scale: int) -> str:
    hr_dir_abs = os.path.abspath(hr_dir)
    dataset_root = os.path.dirname(hr_dir_abs)
    return os.path.join(dataset_root, 'LR_bicubic', f'X{scale}')


def load_or_generate_lr(hr: np.ndarray, hr_path: str, lr_dir: str, scale: int, regenerate_lr: bool) -> np.ndarray:
    os.makedirs(lr_dir, exist_ok=True)
    file_name = os.path.basename(hr_path)
    lr_path = os.path.join(lr_dir, file_name)

    target_h, target_w = hr.shape[:2]
    expected_h, expected_w = target_h // scale, target_w // scale

    if (not regenerate_lr) and os.path.isfile(lr_path):
        lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        if lr is not None and lr.shape[0] == expected_h and lr.shape[1] == expected_w:
            return lr

    lr = make_lr_from_hr(hr, scale)
    cv2.imwrite(lr_path, lr)
    return lr


def evaluate_one_model(
    model_name: str,
    upsampler,
    hr_images: List[str],
    output_dir: str,
    lr_dir: str,
    scale: int,
    crop_border: int,
    test_y_channel: bool,
    save_vis_num: int,
    regenerate_lr: bool,
    lpips_metric: Optional[LPIPSMetric],
    report_psnr_floor: Optional[float]
) -> Dict[str, Optional[float]]:
    os.makedirs(output_dir, exist_ok=True)
    sr_dir = os.path.join(output_dir, 'sr')
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    rows = []
    psnrs, ssims, lpips_scores = [], [], []

    for idx, hr_path in enumerate(hr_images):
        name = os.path.basename(hr_path)
        stem, _ = os.path.splitext(name)

        hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if hr is None:
            print(f'[WARN] unreadable image: {hr_path}')
            continue

        hr = mod_crop(hr, scale)
        lr = load_or_generate_lr(hr, hr_path, lr_dir, scale, regenerate_lr)

        try:
            sr, _ = upsampler.enhance(lr, outscale=scale)
        except Exception as e:
            print(f'[ERROR] enhance failed on {name}: {e}')
            continue

        h, w = hr.shape[:2]
        if sr.shape[0] != h or sr.shape[1] != w:
            sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_CUBIC)

        sr_path = os.path.join(sr_dir, f'{stem}_sr.png')
        cv2.imwrite(sr_path, sr)

        psnr = calculate_psnr(sr, hr, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
        ssim = calculate_ssim(sr, hr, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
        lpips_score = lpips_metric.calculate(sr, hr) if lpips_metric is not None else None

        psnrs.append(float(psnr))
        ssims.append(float(ssim))
        if lpips_score is not None:
            lpips_scores.append(float(lpips_score))

        is_outlier = bool(report_psnr_floor is not None and psnr < report_psnr_floor)
        row = [name, f'{psnr:.4f}', f'{ssim:.4f}']
        if lpips_score is not None:
            row.append(f'{lpips_score:.4f}')
        row.append('1' if is_outlier else '0')
        rows.append(row)

        if idx < save_vis_num:
            vis_path = os.path.join(vis_dir, f'{stem}_triplet.png')
            save_triplet_visual(lr, sr, hr, vis_path)

        print_msg = f'[{model_name}] {name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}'
        if lpips_score is not None:
            print_msg += f', LPIPS={lpips_score:.4f}'
        print(print_msg)

    if len(psnrs) == 0:
        raise RuntimeError(f'No valid images evaluated for model {model_name}')

    avg_psnr = float(np.mean(psnrs))
    avg_ssim = float(np.mean(ssims))
    avg_lpips = float(np.mean(lpips_scores)) if lpips_scores else None

    kept_indices = list(range(len(psnrs)))
    if report_psnr_floor is not None:
        kept_indices = [i for i, v in enumerate(psnrs) if v >= report_psnr_floor]
    dropped_count = len(psnrs) - len(kept_indices)

    avg_psnr_filtered = None
    avg_ssim_filtered = None
    avg_lpips_filtered = None
    if report_psnr_floor is not None and kept_indices:
        avg_psnr_filtered = float(np.mean([psnrs[i] for i in kept_indices]))
        avg_ssim_filtered = float(np.mean([ssims[i] for i in kept_indices]))
        if lpips_scores:
            avg_lpips_filtered = float(np.mean([lpips_scores[i] for i in kept_indices]))

    csv_path = os.path.join(output_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['image', 'psnr', 'ssim'] + (['lpips'] if lpips_metric is not None else []) + ['is_outlier_by_psnr_floor']
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow([])
        summary = ['AVG', f'{avg_psnr:.4f}', f'{avg_ssim:.4f}']
        if avg_lpips is not None:
            summary.append(f'{avg_lpips:.4f}')
        summary.append('')
        writer.writerow(summary)
        if report_psnr_floor is not None:
            floor_summary = [f'AVG(psnr>={report_psnr_floor:g})', '', '']
            if avg_psnr_filtered is not None and avg_ssim_filtered is not None:
                floor_summary[1] = f'{avg_psnr_filtered:.4f}'
                floor_summary[2] = f'{avg_ssim_filtered:.4f}'
            if lpips_metric is not None:
                floor_summary.append(f'{avg_lpips_filtered:.4f}' if avg_lpips_filtered is not None else '')
            floor_summary.append(f'dropped={dropped_count}')
            writer.writerow(floor_summary)

    txt_path = os.path.join(output_dir, 'summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Images: {len(psnrs)}\n')
        f.write(f'Average PSNR: {avg_psnr:.4f}\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')
        if avg_lpips is not None:
            f.write(f'Average LPIPS: {avg_lpips:.4f}\n')
        if report_psnr_floor is not None:
            f.write(f'PSNR outlier floor for supplementary stats: {report_psnr_floor:g}\n')
            f.write(f'Dropped images (PSNR < floor): {dropped_count}\n')
            if avg_psnr_filtered is not None and avg_ssim_filtered is not None:
                f.write(f'Filtered Average PSNR: {avg_psnr_filtered:.4f}\n')
                f.write(f'Filtered Average SSIM: {avg_ssim_filtered:.4f}\n')
                if avg_lpips_filtered is not None:
                    f.write(f'Filtered Average LPIPS: {avg_lpips_filtered:.4f}\n')

    print(f'\n[{model_name}] AVG PSNR={avg_psnr:.4f}, AVG SSIM={avg_ssim:.4f}')
    if avg_lpips is not None:
        print(f'[{model_name}] AVG LPIPS={avg_lpips:.4f}')
    if report_psnr_floor is not None:
        print(f'[{model_name}] Supplementary filtered stats use PSNR floor={report_psnr_floor:g}, dropped={dropped_count}')
    print()
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_lpips': avg_lpips,
        'avg_psnr_filtered': avg_psnr_filtered,
        'avg_ssim_filtered': avg_ssim_filtered,
        'avg_lpips_filtered': avg_lpips_filtered,
        'dropped_count': float(dropped_count),
        'kept_count': float(len(kept_indices)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SR methods and export paper-ready results.')
    parser.add_argument('--hr-dir', type=str, required=True, help='HR(GT) image folder.')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['RealESRGAN_x4plus', 'realesr-general-x4v3'],
        choices=['Bicubic', 'ESRGAN_x4', 'RealESRNet_x4plus', 'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'realesr-general-x4v3']
    )
    parser.add_argument('--output', type=str, default='evaluation_results')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--crop-border', type=int, default=4)
    parser.add_argument('--test-y-channel', action='store_true', help='Compute PSNR/SSIM on Y channel.')
    parser.add_argument('--calc-lpips', action='store_true', help='Compute LPIPS (lower is better).')
    parser.add_argument('--save-vis-num', type=int, default=20, help='How many triplet visuals to save.')
    parser.add_argument('--lr-dir', type=str, default=None, help='LR cache folder. Default: sibling path of HR, e.g. .../LR_bicubic/X4')
    parser.add_argument('--regenerate-lr', action='store_true', help='Force regenerate LR images even if cached LR exists.')
    parser.add_argument('--report-psnr-floor', type=float, default=None, help='Supplementary outlier floor. Keep full-set metrics, and additionally report filtered metrics where PSNR >= floor.')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only.')
    parser.add_argument('--denoise-strength', type=float, default=0.5, help='Only for realesr-general-x4v3.')
    args = parser.parse_args()

    if args.calc_lpips and lpips is None:
        raise ImportError('lpips is required for --calc-lpips. Please run: python -m pip install lpips')

    if not os.path.isdir(args.hr_dir):
        raise FileNotFoundError(f'HR dir not found: {args.hr_dir}')

    hr_images = list_images(args.hr_dir)
    if len(hr_images) == 0:
        raise RuntimeError(f'No images found in {args.hr_dir}')

    os.makedirs(args.output, exist_ok=True)
    lr_dir = args.lr_dir if args.lr_dir else get_default_lr_dir(args.hr_dir, args.scale)
    os.makedirs(lr_dir, exist_ok=True)
    print(f'LR cache dir: {lr_dir}')

    lpips_metric = LPIPSMetric(use_cpu=args.cpu) if args.calc_lpips else None
    compare_rows = []

    for model_name in args.models:
        print(f'\n===== Evaluating {model_name} =====')
        upsampler = init_model(model_name, use_cpu=args.cpu, denoise_strength=args.denoise_strength)
        model_out = os.path.join(args.output, model_name)
        result = evaluate_one_model(
            model_name=model_name,
            upsampler=upsampler,
            hr_images=hr_images,
            output_dir=model_out,
            lr_dir=lr_dir,
            scale=args.scale,
            crop_border=args.crop_border,
            test_y_channel=args.test_y_channel,
            save_vis_num=args.save_vis_num,
            regenerate_lr=args.regenerate_lr,
            lpips_metric=lpips_metric,
            report_psnr_floor=args.report_psnr_floor
        )
        row = [model_name, f'{result["avg_psnr"]:.4f}', f'{result["avg_ssim"]:.4f}']
        if result['avg_lpips'] is not None:
            row.append(f'{result["avg_lpips"]:.4f}')
        if args.report_psnr_floor is not None:
            row.extend([
                f'{result["avg_psnr_filtered"]:.4f}' if result['avg_psnr_filtered'] is not None else '',
                f'{result["avg_ssim_filtered"]:.4f}' if result['avg_ssim_filtered'] is not None else '',
                f'{result["avg_lpips_filtered"]:.4f}' if result['avg_lpips_filtered'] is not None else '',
                str(int(result['kept_count'])) if result['kept_count'] is not None else '',
                str(int(result['dropped_count'])) if result['dropped_count'] is not None else ''
            ])
        compare_rows.append(row)

    compare_csv = os.path.join(args.output, 'model_compare.csv')
    with open(compare_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['model', 'avg_psnr', 'avg_ssim'] + (['avg_lpips'] if args.calc_lpips else [])
        if args.report_psnr_floor is not None:
            header += ['avg_psnr_filtered', 'avg_ssim_filtered', 'avg_lpips_filtered', 'kept_count', 'dropped_count']
        writer.writerow(header)
        writer.writerows(compare_rows)

    compare_txt = os.path.join(args.output, 'model_compare.txt')
    with open(compare_txt, 'w', encoding='utf-8') as f:
        f.write('Model Comparison\n')
        f.write('================\n')
        for row in compare_rows:
            line = f'{row[0]}: PSNR={row[1]}, SSIM={row[2]}'
            if len(row) > 3:
                line += f', LPIPS={row[3]}'
            f.write(line + '\n')

    print('\nDone. Results saved to:', args.output)


if __name__ == '__main__':
    main()
