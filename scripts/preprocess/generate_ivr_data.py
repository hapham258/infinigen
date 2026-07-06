import argparse
import shutil
from tqdm import tqdm
from pathlib import Path
import numpy as np
import OpenImageIO as oiio
import imageio.v3 as imageio

KEEPING_LIST = [
    "camview",
    "Depth",
    "DiffCol",
    "DiffDir",
    "DiffInd",
    "Emit",
    "Env",
    "Flow",
    "Image",
    "imu_tum",
    "SurfaceNormal",
]


def copy_tree(src: Path, dst: Path):
    """Copy the entire directory."""
    if not src.exists():
        print(f"[Skip] {src}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[Copied] {src}")


def copy_files_with_extension(src: Path, dst: Path, extension: str):
    """
    Copy only files with the given extension.
    """
    if not src.exists():
        print(f"[Skip] {src}")
        return
    extension = extension.lower()
    for file in src.rglob(f"*{extension}"):
        rel = file.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if extension == ".exr":
            img = read_exr(file)
            write_exr(out, img, compression="piz")
        elif extension == ".npy":
            arr = np.load(file)
            write_npz(out.with_suffix(".npz"), arr)
        else:
            shutil.copy2(file, out)
    if extension == ".exr":
        print(f"[Compressed to float16 {extension}] {src}")
    elif extension == ".npy":
        print(f"[Compressed to float16 {extension}] {src}")
    else:
        print(f"[Copied {extension}] {src}")


def convert_flow_npy(src: Path, dst: Path):
    """Convert Infinigen .npy flow to compressed float16 .npz."""
    if not src.exists():
        print(f"[Skip] {src}")
        return
    for file in src.rglob("*.npy"):
        flow = np.load(file)[..., 1:3].astype(np.float16)
        rel = file.relative_to(src).with_suffix(".npz")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, flow)
    print(f"[Converted] {src}")


def read_exr(path: Path) -> np.ndarray:
    """Read an EXR image into a float32 NumPy array."""
    buf = oiio.ImageBuf(str(path))
    img = buf.get_pixels()
    if img.size == 0:
        raise RuntimeError(f"Failed to read EXR: {path}")
    return img


def write_exr(path: Path, img: np.ndarray, compression: str = "piz"):
    """Write a NumPy array to an EXR image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(img, dtype=np.float16)
    spec = oiio.ImageSpec(
        img.shape[1],  # width
        img.shape[0],  # height
        img.shape[2],  # channels
        oiio.FLOAT,
    )
    spec.attribute("compression", compression)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"Failed to create EXR: {path}")
    try:
        if not out.open(str(path), spec):
            raise RuntimeError(f"Failed to open EXR: {path}")
        out.write_image(img)
    finally:
        out.close()


def write_npz(path: Path, arr: np.ndarray, key: str = "data"):
    """Write a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, arr.astype(np.float16))


def combine_diff_shading(diffdir_src: Path, diffind_src: Path, shading_dst: Path):
    """
    Combine DiffDir and DiffInd into DiffShading:
        DiffShading = DiffDir + DiffInd
    """
    #
    if not diffdir_src.exists() or not diffind_src.exists():
        print(f"[Skip] {diffdir_src} or {diffind_src} does not exist.")
        return
    diffdir_files = sorted(diffdir_src.rglob("DiffDir*.exr"))

    #
    for diffdir_file in tqdm(
        diffdir_files,
        desc="Generating DiffShading",
        unit="image",
    ):
        #
        diffind_file = diffind_src / diffdir_file.relative_to(diffdir_src)
        diffind_file = diffind_file.with_name(
            diffdir_file.name.replace("DiffDir", "DiffInd", 1)
        )
        if not diffind_file.exists():
            tqdm.write(f"[Missing] {diffind_file}")
            continue

        #
        try:
            diffdir = read_exr(diffdir_file)
            diffind = read_exr(diffind_file)
        except RuntimeError as e:
            tqdm.write(str(e))
            continue
        shading = diffdir + diffind
        shading_file = shading_dst / diffdir_file.relative_to(diffdir_src)
        shading_file = shading_file.with_name(
            shading_file.name.replace("DiffDir", "DiffShading", 1)
        )
        write_exr(shading_file, shading)
    print(f"[Generated to float16] {shading_dst}")


def compute_residual_confidence(color, diff_reflect, diff_illumi, gamma=2.0):
    """
    Compute a non-diffuse confidence map.

    diffuse pixel -> weight ≈ 0
    mirror        -> weight ≈ 1
    shiny metal   -> intermediate
    """
    eps = 1e-6
    diffuse = diff_reflect * diff_illumi
    res = color - diffuse
    res_intens = np.mean(np.abs(res), axis=-1)
    color_intens = np.mean(color, axis=-1)
    res_conf = res_intens / np.maximum(color_intens, eps)
    res_conf = res_conf**gamma
    res_conf = np.clip(res_conf, 0.0, 1.0)
    return res, res_conf


def generate_residual_confidence(
    image_dir: Path,
    diffcol_dir: Path,
    diffdir_dir: Path,
    diffind_dir: Path,
    residual_dir: Path,
    confidence_dir: Path,
    gamma: float = 2.0,
):
    """
    Generate:
        NonDiffRes  = Image - DiffCol * (DiffDir + DiffInd)
        NonDiffConf = residual confidence map (8-bit PNG)
    """
    image_files = sorted(image_dir.rglob("Image*.exr"))
    for image_file in tqdm(
        image_files,
        desc="Generating NonDiffRes",
        unit="image",
    ):
        #
        rel = image_file.relative_to(image_dir)
        diffcol_file = diffcol_dir / rel
        diffcol_file = diffcol_file.with_name(
            image_file.name.replace("Image", "DiffCol", 1)
        )
        diffdir_file = diffdir_dir / rel
        diffdir_file = diffdir_file.with_name(
            image_file.name.replace("Image", "DiffDir", 1)
        )
        diffind_file = diffind_dir / rel
        diffind_file = diffind_file.with_name(
            image_file.name.replace("Image", "DiffInd", 1)
        )
        if (
            not diffcol_file.exists()
            or not diffdir_file.exists()
            or not diffind_file.exists()
        ):
            tqdm.write(f"[Missing] {rel}")
            continue

        #
        color = read_exr(image_file)
        diff_reflect = read_exr(diffcol_file)
        diff_dir = read_exr(diffdir_file)
        diff_ind = read_exr(diffind_file)
        if (
            color.shape != diff_reflect.shape
            or color.shape != diff_dir.shape
            or color.shape != diff_ind.shape
        ):
            tqdm.write(f"[Shape mismatch] {rel}")
            continue
        diff_illumi = diff_dir + diff_ind
        residual, confidence = compute_residual_confidence(
            color,
            diff_reflect,
            diff_illumi,
            gamma=gamma,
        )

        #
        residual_file = residual_dir / rel
        residual_file = residual_file.with_name(
            residual_file.name.replace("Image", "NonDiffRes", 1)
        )
        write_exr(residual_file, residual)

        #
        confidence_file = confidence_dir / rel
        confidence_file = confidence_file.with_name(
            confidence_file.name.replace("Image", "NonDiffConf", 1)
        ).with_suffix(".png")
        confidence_file.parent.mkdir(parents=True, exist_ok=True)
        confidence_u8 = (confidence * 255).astype(np.uint8)
        imageio.imwrite(confidence_file, confidence_u8)
    print(f"[Generated] {residual_dir}")
    print(f"[Generated] {confidence_dir}")


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", type=Path, help="Source root directory")
    parser.add_argument("dst_root", type=Path, help="Destination root directory")
    parser.add_argument(
        "scene_list", type=Path, help="Text file containing one scene ID per line."
    )
    args = parser.parse_args()

    #
    with args.scene_list.open("r") as f:
        scenes = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    #
    for scene in scenes:
        #
        src_scene_dir = args.src_root / scene
        if not src_scene_dir.exists():
            print(f"[SRC_SKIP] {src_scene_dir} does not exist.")
            continue
        dst_scene_dir = args.dst_root / scene
        if dst_scene_dir.exists():
            print(f"[DST_SKIP] {dst_scene_dir} already exists.")
            continue

        #
        for name in KEEPING_LIST:
            #
            if name in ("DiffDir", "DiffInd"):
                continue

            #
            src_part_dir = src_scene_dir / "frames" / name
            dst_part_dir = dst_scene_dir / "frames" / name
            if name == "Image":
                copy_files_with_extension(src_part_dir, dst_part_dir, ".exr")
                copy_files_with_extension(src_part_dir, dst_part_dir, ".png")
            elif name in ("Depth", "SurfaceNormal"):
                copy_files_with_extension(src_part_dir, dst_part_dir, ".npy")
                copy_files_with_extension(src_part_dir, dst_part_dir, ".png")
            elif name == "DiffCol":
                copy_files_with_extension(src_part_dir, dst_part_dir, ".exr")
            elif name == "Flow":
                convert_flow_npy(src_part_dir, dst_part_dir)
            else:
                copy_tree(src_part_dir, dst_part_dir)

        #
        combine_diff_shading(
            src_scene_dir / "frames" / "DiffDir",
            src_scene_dir / "frames" / "DiffInd",
            dst_scene_dir / "frames" / "DiffShading",
        )

        #
        generate_residual_confidence(
            src_scene_dir / "frames" / "Image",
            src_scene_dir / "frames" / "DiffCol",
            src_scene_dir / "frames" / "DiffDir",
            src_scene_dir / "frames" / "DiffInd",
            dst_scene_dir / "frames" / "NonDiffRes",
            dst_scene_dir / "frames" / "NonDiffConf",
        )
    print("\nDone!")
