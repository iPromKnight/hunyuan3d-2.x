import os
import sys
import random
import shutil
import time
import uuid
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from hy3dgen.shapegen.utils import logger
import threading
import gc

torch.set_grad_enabled(False)
try: torch.set_flush_denormal(True)
except Exception: pass
pipeline_lock = threading.Lock()

MAX_SEED = int(1e7)
DEFAULT_EXPORT_FMT = "stl"
AUTO_LOADED = False
MODEL_CHOICES = {
    "v2.0 (standard)": {
        "model_path": "tencent/Hunyuan3D-2",
        "subfolder":  "hunyuan3d-dit-v2-0",
    },
    "v2.0 (turbo)": {
        "model_path": "tencent/Hunyuan3D-2",
        "subfolder":  "hunyuan3d-dit-v2-0-turbo",
    },
    "v2.1 (standard)": {
        "model_path": "tencent/Hunyuan3D-2.1",
        "subfolder": "hunyuan3d-dit-v2-1"
    }
}
DEFAULT_MODEL_KEY = "v2.0 (turbo)"
BACKEND = None
i23d_worker = None

if sys.platform == "darwin":
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*multidimensional indexing is deprecated.*",
        category=UserWarning
    )

class _Backend:
    def __init__(self, family: str):
        if family == "21":
            from hy3dshape.pipelines import (
                Hunyuan3DDiTFlowMatchingPipeline as _Pipe,
                export_to_trimesh as _export_to_trimesh,
            )
            try:
                from hy3dshape import FaceReducer as _FR, FloaterRemover as _FL, DegenerateFaceRemover as _DF
            except Exception:
                from hy3dshape.post import FaceReducer as _FR, FloaterRemover as _FL, DegenerateFaceRemover as _DF

            try:
                from hy3dshape.rembg import BackgroundRemover as _BR
            except Exception:
                from hy3dshape.utils.rembg import BackgroundRemover as _BR

            self.Hunyuan3DDiTFlowMatchingPipeline = _Pipe
            self.export_to_trimesh = _export_to_trimesh
            self.FaceReducer = _FR
            self.FloaterRemover = _FL
            self.DegenerateFaceRemover = _DF
            self.BackgroundRemover = _BR
            self.use_safetensors = False  # 2.1 often uses ckpt
        else:
            from hy3dgen.shapegen import (
                Hunyuan3DDiTFlowMatchingPipeline as _Pipe,
                FaceReducer as _FR, FloaterRemover as _FL,
                DegenerateFaceRemover as _DF,
                pipelines as _pip
            )
            from hy3dgen.rembg import BackgroundRemover as _BR
            self.Hunyuan3DDiTFlowMatchingPipeline = _Pipe
            self.export_to_trimesh = _pip.export_to_trimesh
            self.FaceReducer = _FR
            self.FloaterRemover = _FL
            self.DegenerateFaceRemover = _DF
            self.BackgroundRemover = _BR
            self.use_safetensors = True

def _log_mem(label: str):
    try:
        if args.device == "cuda" and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            print(f"[mem][{label}] cuda_allocated_mb={alloc:.1f} reserved_mb={reserv:.1f}")
        elif args.device == "mps" and getattr(torch, "mps", None):
            # ru_maxrss units: bytes on macOS, KiB on Linux
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss_mb = rss / (1024**2)
            else:
                rss_mb = rss / 1024  # KiB -> MiB
            print(f"[mem][{label}] rss_mb={rss_mb:.1f}")
    except Exception as e:
        print(f"[mem][{label}] error: {e}")

def _mps_sync():
    try:
        if args.device == "mps" and getattr(torch, "mps", None):
            torch.mps.synchronize()
    except Exception:
        pass

def _reset_backend_state():
    global BACKEND, i23d_worker
    global rmbg_worker, floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker
    BACKEND = None
    i23d_worker = None
    rmbg_worker = None
    floater_remove_worker = None
    degenerate_face_remove_worker = None
    face_reduce_worker = None
    _empty_device_cache()

def _select_backend(model_path: str, subfolder: str) -> _Backend:
    return _Backend("21" if _is_21(model_path, subfolder) else "20")

def get_example_img_list() -> List[str]:
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))

def get_example_txt_list() -> List[str]:
    print('Loading example txt list ...')
    p = Path('./assets/example_prompts.txt')
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]

def get_example_mv_list() -> List[List[Optional[str]]]:
    print('Loading example mv list ...')
    root = Path('./assets/example_mv_images')
    if not root.exists():
        return []
    mv_list: List[List[Optional[str]]] = []
    for mv_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        view_list: List[Optional[str]] = []
        for view in ['front', 'back', 'left', 'right']:
            path = mv_dir / f'{view}.png'
            view_list.append(str(path) if path.exists() else None)
        mv_list.append(view_list)
    return mv_list

def gen_save_folder(max_size: int = 200) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    with os.scandir(SAVE_DIR) as it:
        dirs = [d for d in it if d.is_dir()]
    if len(dirs) > max_size:
        dirs_sorted = sorted(dirs, key=lambda d: d.stat().st_ctime)
        target = int(max_size * 0.8)
        for d in dirs_sorted[:max(0, len(dirs) - target)]:
            shutil.rmtree(d.path, ignore_errors=True)
    new_folder = os.path.join(SAVE_DIR, uuid.uuid4().hex)
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def export_mesh(mesh: trimesh.Trimesh, save_folder: str, type: str = 'stl') -> str:
    path = os.path.join(save_folder, f'white_mesh.{type}')
    if type == 'stl':
        mesh.export(path, file_type='stl')
    elif type in ('glb', 'obj'):
        mesh.export(path, include_normals=False)
    else:
        mesh.export(path)
    return path

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    return random.randint(0, MAX_SEED) if randomize_seed else seed

@lru_cache(maxsize=2)
def load_template(textured: bool) -> str:
    # We only use the white_mesh (non-textured) template now, but keep switch for future extensibility.
    template_name = './assets/modelviewer-template.html'
    return Path(template_name).read_text(encoding='utf-8')

def build_model_viewer_html(save_folder: str, height: int = 650, width: int = 790) -> str:
    # Fixed non-textured path
    related_path = "./white_mesh.glb"
    template_html = load_template(textured=False)
    # Slight offset for toolbar
    offset = 10
    output_html_path = os.path.join(save_folder, 'white_mesh.html')
    html = (template_html
            .replace('#height#', f'{height - offset}')
            .replace('#width#', f'{width}')
            .replace('#src#', f'{related_path}/'))
    Path(output_html_path).write_text(html, encoding='utf-8')
    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Wrote {output_html_path}, relative path /static/{rel_path}')
    return f"<div style='height: {height}; width: 100%;'>{iframe_tag}</div>"

def _empty_device_cache():
    try:
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif args.device == "mps" and getattr(torch, "mps", None):
            torch.mps.empty_cache()
    except Exception:
        pass

def _model_label(model_path: str, subfolder: str) -> str:
    if not model_path or not subfolder:
        return "— none —"
    return f"{model_path}/{subfolder}"

def reload_shapegen(model_key: str):
    """
    Reinitialize i23d_worker with a different model/subfolder at runtime.
    Returns: (panel_status_html, title_html, num_steps_value, header_label, panel_visibility, inline_status_html)
    """
    global i23d_worker, TURBO_MODE, args
    global BACKEND, rmbg_worker, floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker

    if not model_key or model_key not in MODEL_CHOICES:
        err = "<div style='color:#b91c1c'>Invalid model selection.</div>"
        return gr.update(value=err), gr.update(), gr.update(), gr.update(value=_model_label(args.model_path, args.subfolder)), gr.update(visible=True), gr.update(value=err)

    cfg = MODEL_CHOICES[model_key]
    new_model_path = cfg["model_path"]
    new_subfolder  = cfg["subfolder"]

    if (
        i23d_worker is not None 
        and BACKEND is not None
        and args.model_path == new_model_path 
        and args.subfolder == new_subfolder
    ):
        default_steps = 8 if "turbo" in (new_subfolder or "") else 30
        title = 'Image to 3D (1–4 Views)' if MV_MODE else 'High Resolution 3D Shape Generation'
        if "turbo" in (new_subfolder or ""):
            title = title.replace(':', '-Turbo: Fast ')
        title_html = f"""
        <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
        {title}
        </div>
        <div align="center">PromGen</div>
        """
        ok_inline = gr.update(
            value=f"<div style='margin-top:6px;color:#065f46;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:6px 10px;'>✅ Model ready: <code style='color:#065f46;'>{new_model_path}/{new_subfolder}</code> (already loaded)</div>"
        )
        return (
            gr.update(value="<div style='color:#065f46'>Already loaded.</div>"),
            gr.update(value=title_html),
            gr.update(value=default_steps),
            gr.update(value=_model_label(new_model_path, new_subfolder)),
            gr.update(visible=False),
            ok_inline,
        )

    status_lines = []
    t0 = time.time()

    with pipeline_lock:
        _reset_backend_state()
        status_lines.append(f"Loading {new_model_path}/{new_subfolder} on {args.device}…")

        try:
            # pick family (may import hy3dshape/hy3dgen)
            BACKEND = _select_backend(new_model_path, new_subfolder)

            # build helpers AFTER we know backend is importable
            rmbg_worker = BACKEND.BackgroundRemover()
            floater_remove_worker = BACKEND.FloaterRemover()
            degenerate_face_remove_worker = BACKEND.DegenerateFaceRemover()
            face_reduce_worker = BACKEND.FaceReducer()

            # load the main pipeline
            i23d_worker = BACKEND.Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                new_model_path,
                subfolder=new_subfolder,
                use_safetensors=BACKEND.use_safetensors,
                device=args.device,
            )
            if args.enable_flashvdm:
                i23d_worker.enable_flashvdm(mc_algo=args.mc_algo)
            if args.compile:
                i23d_worker.compile()

            # success: persist selection
            args.model_path = new_model_path
            args.subfolder  = new_subfolder
            TURBO_MODE = "turbo" in (new_subfolder or "")
            AUTO_LOADED = True

            elapsed = time.time() - t0
            status_lines.append(f"✅ Loaded in {elapsed:.1f}s")

        except Exception as e:
            # hard reset so next selection starts clean
            _reset_backend_state()
            joined = "<br>".join(status_lines + [f"❌ Load failed: {e}"])
            return (
                gr.update(value=f"<div style='color:#b91c1c'>{joined}</div>"),
                gr.update(), gr.update(),
                gr.update(value=_model_label(args.model_path, args.subfolder)),
                gr.update(visible=True),  # keep panel open
                gr.update(value=f"<div style='color:#b91c1c'>❌ Load failed: {e}</div>")
            )

    # Title + defaults
    if MV_MODE:
        title = 'Image to 3D (1–4 Views)'
    elif 'mini' in new_subfolder:
        title = '0.6B Image→Shape'
    else:
        title = 'High Resolution 3D Shape Generation'
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    default_steps = 5 if TURBO_MODE else 30

    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
      {title}
    </div>
    <div align="center">PromGen</div>
    """

    joined = "<br>".join(status_lines)
    panel_status = gr.update(value=f"<div style='color:#065f46'>{joined}</div>")
    inline_status = gr.update(value=f"<div style='margin-top:6px;color:#065f46;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:6px 10px;'>✅ Model ready: <code style='color:#065f46;'>{new_model_path}/{new_subfolder}</code></div>")
    _log_mem("reload_shapegen:loaded")
    return (
        panel_status,                      # status inside panel
        gr.update(value=title_html),       # title update
        gr.update(value=default_steps),    # new default steps
        gr.update(value=_model_label(new_model_path, new_subfolder)),  # label under button
        gr.update(visible=False),          # close the panel
        inline_status,                     # show success under the button
    )

# -----------------------
# Core generation
# -----------------------
def _is_21(model_path: str, subfolder: str) -> bool:
    return ("2.1" in (model_path or "")) or ("v2-1" in (subfolder or ""))

def _has_transparency(pil_img) -> bool:
    if pil_img.mode in ('LA', 'RGBA'):
        alpha = pil_img.getchannel('A')
        return alpha.getextrema()[0] < 255
    return False

def _prepare_image_inputs(
    caption,
    image,
    mv_image_front,
    mv_image_back,
    mv_image_left,
    mv_image_right,
    check_box_rembg: bool
):
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Please provide either a caption or an image.")
    if MV_MODE:
        if not any([mv_image_front, mv_image_back, mv_image_left, mv_image_right]):
            raise gr.Error("Please provide at least one view image.")
        image_dict: Dict[str, any] = {}
        if mv_image_front: image_dict['front'] = mv_image_front
        if mv_image_back:  image_dict['back'] = mv_image_back
        if mv_image_left:  image_dict['left'] = mv_image_left
        if mv_image_right: image_dict['right'] = mv_image_right
        return image_dict
    return image

@torch.inference_mode()
def _gen_shape(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 1234,
    octree_resolution: int = 256,
    check_box_rembg: bool = False,
    num_chunks: int = 8000,
    randomize_seed: bool = False,
) -> Tuple[trimesh.Trimesh, any, str, Dict, int]:
    if BACKEND is None or i23d_worker is None:
        raise gr.Error("Internal: backend not initialized. Click “Load model” and try again.")
    
    _log_mem("_gen_shape:begin")
    image = _prepare_image_inputs(caption, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right, check_box_rembg)

    seed = int(randomize_seed_fn(seed, randomize_seed))
    octree_resolution = int(octree_resolution)
    if caption:
        print('prompt is', caption)

    save_folder = gen_save_folder()
    stats: Dict = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        },
        'time': {}
    }

    # Text → image (optional)
    if image is None:
        t0 = time.time()
        try:
            image = t2i_worker(caption)
        except Exception:
            raise gr.Error("Text-to-Image disabled. Start with --enable_t23d to use it.")
        stats['time']['text2image'] = time.time() - t0

    # Rembg if requested or image lacks alpha
    def _rembg_one(img):
        # Only run if explicitly requested OR image is fully opaque
        if check_box_rembg and (img.mode not in ('LA', 'RGBA') or not _has_transparency(img)):
            return rmbg_worker(img.convert('RGB'))
        return img

    t0 = time.time()
    if MV_MODE:
        for k, v in list(image.items()):
            image[k] = _rembg_one(v)
    else:
        image = _rembg_one(image)
    stats['time']['remove background'] = time.time() - t0

    # Shape generation
    t0 = time.time()
    g = torch.Generator(device=args.device if args.device in ('cuda', 'mps') else 'cpu').manual_seed(seed)
    try:
        outputs = i23d_worker(
            image=image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=g,
            octree_resolution=int(octree_resolution),
            num_chunks=int(num_chunks),
            output_type='mesh'
        )
    except KeyboardInterrupt:
        raise gr.Error("Generation cancelled.")
    except RuntimeError as e:
        msg = str(e).lower()
        if 'memory' in msg or 'resource' in msg or 'mps' in msg:
            raise gr.Error("Out of memory. Lower Octree Resolution (e.g. 196), reduce Inference Steps, "
                        "or decrease Number of Chunks.")
        raise
    _mps_sync()
    stats['time']['shape generation'] = time.time() - t0
    logger.info("---Shape generation takes %.3f seconds ---", stats['time']['shape generation'])

    # Convert to trimesh
    t0 = time.time()
    exported = BACKEND.export_to_trimesh(outputs)
    # free the big tensors ASAP
    try:
        del outputs
    except Exception:
        pass
    _mps_sync()

    # Normalize to a single Trimesh
    mesh = exported[0] if isinstance(exported, (list, tuple)) else exported
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            raise gr.Error("Generated scene is empty. Try more steps or a higher octree resolution.")
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        raise gr.Error(f"Unexpected export type: {type(mesh)}")

    if mesh.faces is None or len(mesh.faces) == 0 or mesh.vertices is None or len(mesh.vertices) < 3:
        raise gr.Error("Generated mesh is empty/degenerate. Increase steps or octree resolution.")

    stats['time']['export to trimesh'] = time.time() - t0

    try:
        _ = mesh.face_normals  # builds cache
    except Exception:
        # attempt to fix then retry once
        try:
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_faces()
            _ = mesh.face_normals
        except Exception:
            # we'll continue; exporters can still succeed without the cache
            pass

    stats['number_of_faces'] = int(mesh.faces.shape[0])
    stats['number_of_vertices'] = int(mesh.vertices.shape[0])

    main_image = image if not MV_MODE else image.get('front')

    gc.collect()
    _empty_device_cache()
    _log_mem("_gen_shape:end")

    return mesh, main_image, save_folder, stats, seed

@torch.inference_mode()
def shape_generation(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    if i23d_worker is None:
        raise gr.Error("No model loaded yet. Click “Change model”, pick a model, then click “Load model”.")
    _log_mem("shape_generation:begin")
    try:
        start_time_0 = time.time()
        mesh, image, save_folder, stats, seed = _gen_shape(
            caption=caption,
            image=image,
            mv_image_front=mv_image_front,
            mv_image_back=mv_image_back,
            mv_image_left=mv_image_left,
            mv_image_right=mv_image_right,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            octree_resolution=octree_resolution,
            check_box_rembg=check_box_rembg,
            num_chunks=num_chunks,
            randomize_seed=randomize_seed,
        )

        # Optional post-processing hooks (cheap sanity clean)
        t1 = time.time()
        mesh = floater_remove_worker(mesh)
        stats['time']['floater_remove'] = time.time() - t1

        t1 = time.time()
        mesh = degenerate_face_remove_worker(mesh)
        stats['time']['degenerate_remove'] = time.time() - t1

        # --- Export for download (STL, max faces = no simplification) ---
        t1 = time.time()
        path_download = export_mesh(mesh, save_folder, type=DEFAULT_EXPORT_FMT)
        stats['time']['export_stl'] = time.time() - t1

        # Defer GLB export + HTML preview to a chained step for snappier UX
        placeholder = f"<div style='height: {HTML_HEIGHT}px; display:flex; align-items:center; justify-content:center; color:#6b7280;'>Building preview…</div>"

        stats['time']['total'] = time.time() - start_time_0
        mesh.metadata['extras'] = stats

        for k, v in stats['time'].items():
            try:
                print(f"[timing] {k}: {v:.3f}s")
            except Exception:
                pass

        return (
            gr.update(value=path_download),
            placeholder,
            stats,
            seed,
            save_folder,  # hand over to chained preview step
        )
    finally:
        gc.collect()
        _empty_device_cache()
        _log_mem("shape_generation:end")
# -----------------------
# UI / App
# -----------------------
def build_preview(save_folder: Optional[str]):
    try:
        if not save_folder:
            return gr.update()
        _log_mem("build_preview:begin")
        stl_path = os.path.join(save_folder, 'white_mesh.stl')
        if not os.path.exists(stl_path):
            return gr.update()

        mesh = trimesh.load(stl_path, process=False)

        preview_mesh = mesh
        try:
            target = max(20000, int(len(mesh.faces) * 0.2))
            preview_mesh = face_reduce_worker(mesh.copy(), target)
        except Exception:
            preview_mesh = mesh  # fall back

        _ = export_mesh(preview_mesh, save_folder, type='glb')
        html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)

        try:
            del preview_mesh
            del mesh
        except Exception:
            pass

        return html
    except Exception:
        return gr.update()
    finally:
        gc.collect()
        _empty_device_cache()
        _log_mem("build_preview:end")

def autoload_on_start(model_key: str):
    """Run once per server lifetime; no-op on refresh or if model is already loaded."""
    global AUTO_LOADED
    if AUTO_LOADED or i23d_worker is not None:
        # No changes; keep panel closed and show current label (if any)
        return (
            gr.update(value=""),                  # panel_status
            gr.update(),                          # title_html_box (unchanged)
            gr.update(),                          # num_steps (unchanged)
            gr.update(value=_model_label(args.model_path, args.subfolder)),  # current model label
            gr.update(visible=False),             # keep panel hidden
            gr.update(value=""),                  # inline status (no message)
        )
    return reload_shapegen(model_key)

def build_app():
    title = 'High Resolution 3D Shape Generation'
    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
    {title}
    </div>
    <div align="center">PromGen</div>
    <div align="center" style="margin-top:6px;color:#6b7280">
    Active model: <code>{_model_label(args.model_path, args.subfolder)}</code>
    </div>
    """

    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) { max-width: 1280px; }
    .mv-image button .wrap { font-size: 10px; }
    .mv-image .icon-wrap { width: 20px; }
    /* Kill top-right navbar (Use via API / Settings) */
    header, [data-testid="block-navbar"], [data-testid="navbar"], .navbar { display: none !important; }
    /* Kill footer (Built with Gradio) */
    footer, [data-testid="footer"] { display: none !important; }
    /* Optional: hide any leftover portals Gradio uses for global UI */
    #component-portal { display: none !important; }
    """

    with gr.Blocks(theme=gr.themes.Base(), title='PromGen', analytics_enabled=False, css=custom_css) as demo:
        # demo.queue(default_concurrency_limit=1, max_size=4)
        save_folder_state = gr.State()

        # Dynamic title stays just below the header
        title_html_box = gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        # Inline model switcher (below the prompt tabs)
                        with gr.Group():
                            with gr.Row():
                                current_model_html = gr.HTML(
                                    f"<div style='padding:6px 10px;border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb;'>"
                                    f"Model: <code>{_model_label(args.model_path, args.subfolder)}</code>"
                                    f"</div>"
                                )
                                btn_open_model = gr.Button("Change model", variant="secondary", min_width=120)

                            # success/error message shown *outside* the panel after load
                            load_status_inline = gr.HTML("")  

                            # Hidden panel that acts like a modal
                            with gr.Group(visible=False, elem_id="change_model_panel") as change_model_panel:
                                with gr.Column():
                                    model_dropdown = gr.Dropdown(
                                        label="Select model",
                                        choices=list(MODEL_CHOICES.keys()),
                                    )
                                    with gr.Row():
                                        btn_load_model = gr.Button("Load model", variant="primary")
                                        btn_close_model = gr.Button("Close", variant="secondary")
                                    load_status = gr.HTML("")  # status text lives inside the panel
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)

                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='PromGen will generate an image if enabled.',
                                             info='e.g. A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row():
                    btn = gr.Button(value='Generate Shape', variant='primary', min_width=120)

                with gr.Group():
                    file_out = gr.File(label="Generated Mesh (.stl)")

                with gr.Tab('Advanced Options', id='tab_advanced_options'):
                    with gr.Row():
                        check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100)
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100)
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=1234,
                        min_width=100,
                    )
                    with gr.Row():
                        num_steps = gr.Slider(maximum=100,
                                              minimum=1,
                                              value=30,
                                              step=1, label='Inference Steps')
                        octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution')
                    with gr.Row():
                        cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100, visible=False)
                        num_chunks = gr.Slider(maximum=5_000_000, minimum=1_000, value=8_000,
                                               label='Number of Chunks', min_width=100, visible=False)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=2 if not MV_MODE else 3):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery', visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)
                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I and not MV_MODE) as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label=None, examples_per_page=18)
                    with gr.Tab('MultiView to 3D Gallery', id='tab_mv_gallery', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            gr.Examples(examples=example_mvs,
                                        inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right],
                                        label=None, examples_per_page=6)

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        if HAS_T2I:
            tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        model_dropdown.value = DEFAULT_MODEL_KEY
        auto_model_key = gr.State(DEFAULT_MODEL_KEY)
        demo.load(
            reload_shapegen,
            inputs=[auto_model_key],
            outputs=[load_status, title_html_box, num_steps, current_model_html, change_model_panel, load_status_inline],
        )

        # Open panel
        btn_open_model.click(
            lambda: gr.update(visible=True),
            outputs=[change_model_panel],
        )

        # Close panel
        btn_close_model.click(
            lambda: gr.update(visible=False),
            outputs=[change_model_panel],
        )

        # Load model (updates: panel status, title, num_steps, label under button, closes panel, and sets inline status)
        btn_load_model.click(
            reload_shapegen,
            inputs=[model_dropdown],
            outputs=[load_status, title_html_box, num_steps, current_model_html, change_model_panel, load_status_inline],
        )

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed
            ],
            outputs=[file_out, html_gen_mesh, stats, seed, save_folder_state]
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            build_preview,
            inputs=[save_folder_state],
            outputs=[html_gen_mesh],
        )

    return demo

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default=None, help="cpu|cuda|mps (auto if None)")
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true', default=True)
    args = parser.parse_args()
    args.model_path = ""
    args.subfolder  = ""

    print(f"Args: {args}")

    # Device auto-detect for macs
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Using device: {args.device}")

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = False
    TURBO_MODE = False

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to PromGen!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']  # kept for potential future export UI

    # Optional T2I
    HAS_T2I = False
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=args.device)
        HAS_T2I = True

    # Shape generation pipeline
    BACKEND = None
    rmbg_worker = None
    floater_remove_worker = None
    degenerate_face_remove_worker = None
    face_reduce_worker = None
    i23d_worker = None

    # FastAPI + static for previews
    app = FastAPI()
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    
    for item in static_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    
    # Env maps for modelviewer
    shutil.copytree('./assets/env_maps', static_dir / 'env_maps', dirs_exist_ok=True)

    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)