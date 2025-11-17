"""
Module B: Mesh post-processing pipeline for NeRF / instant-ngp exports
---------------------------------------------------------------------

功能：
- 从 OBJ/PLY 读取 mesh
- Laplacian 平滑
- 填洞
- 网格简化（面数减少）
- 自动对齐（惯性主轴对齐世界坐标）
- 归一化到单位包围盒并居中
- 导出处理后的 mesh

用法示例（命令行）：
    python mesh_moduleB.py ^
        --input "D:/nerf_datasets/blender/nerf_synthetic/lego/transforms_train_base.obj" ^
        --output "D:/nerf_datasets/blender/nerf_synthetic/lego/lego_processed.obj" ^
        --target_faces 5000 ^
        --smooth_iter 5
"""

import argparse
import os
import numpy as np
import trimesh
from trimesh.smoothing import filter_laplacian


def print_mesh_info(title: str, mesh: trimesh.Trimesh) -> None:
    """打印 mesh 的基本统计信息，便于写到报告里。"""
    bbox = mesh.bounding_box.extents
    print(f"\n=== {title} ===")
    print(f"Vertices : {len(mesh.vertices)}")
    print(f"Faces    : {len(mesh.faces)}")
    print(f"Bounds   : min {mesh.bounds[0]}, max {mesh.bounds[1]}")
    print(f"Extents  : {bbox} (x, y, z)")


def laplacian_smoothing(mesh: trimesh.Trimesh,
                        iterations: int = 5,
                        lamb: float = 0.5) -> trimesh.Trimesh:
    """
    对 mesh 做 Laplacian 平滑（不会改变拓扑，只减少高频噪声）。
    iterations: 迭代次数，越大越平滑，也更容易"塌陷"
    lamb      : 步长，0.3~0.6 比较常用
    """
    mesh = mesh.copy()
    filter_laplacian(mesh, lamb=lamb, iterations=iterations)
    return mesh


def fill_mesh_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    使用 trimesh 的 repair 工具填补拓扑洞。
    对于 NeRF 提取出来的 mesh，常会有边界小洞。
    """
    mesh = mesh.copy()
    before = mesh.euler_number
    trimesh.repair.fill_holes(mesh)
    after = mesh.euler_number
    print(f"Filled holes: Euler number {before} -> {after}")
    return mesh


def decimate_mesh_safe(mesh: trimesh.Trimesh,
                      target_faces: int = 5000) -> trimesh.Trimesh:
    """
    安全地简化网格，使用多种方法尝试直到成功。
    """
    mesh = mesh.copy()
    current_faces = len(mesh.faces)
    
    if target_faces >= current_faces:
        print(f"[Decimation] target_faces={target_faces} "
              f">= current_faces={current_faces}, skip decimation.")
        return mesh

    print(f"[Decimation] {current_faces} -> {target_faces} faces ...")
    
    # 方法1: 尝试使用 quadric decimation
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        print("[Decimation] Success with quadric decimation")
        return simplified
    except Exception as e1:
        print(f"[Decimation] Quadric decimation failed: {e1}")
    
    # 方法2: 尝试使用比例参数调用 quadric decimation
    try:
        ratio = target_faces / current_faces
        # 有些版本的 trimesh 支持比例参数
        simplified = mesh.simplify_quadric_decimation(ratio)
        print("[Decimation] Success with ratio-based quadric decimation")
        return simplified
    except Exception as e2:
        print(f"[Decimation] Ratio-based quadric decimation failed: {e2}")
    
    # 方法3: 尝试使用不同的简化方法
    try:
        # 使用 trimesh 的简化函数（如果有）
        if hasattr(trimesh, 'simplification'):
            ratio = target_faces / current_faces
            simplified = trimesh.simplification.simplify(mesh, ratio)
            print("[Decimation] Success with trimesh.simplification")
            return simplified
    except Exception as e3:
        print(f"[Decimation] Trimesh simplification failed: {e3}")
    
    # 方法4: 如果所有方法都失败，使用分步简化
    print("[Decimation] Using step-by-step simplification")
    current_mesh = mesh
    step_target = max(target_faces, current_faces // 2)
    
    while current_faces > target_faces * 1.1:  # 留10%余量
        try:
            ratio = step_target / current_faces
            # 最后一次尝试使用 quadric decimation
            if step_target <= target_faces * 1.1:
                current_mesh = current_mesh.simplify_quadric_decimation(target_faces)
                break
            else:
                current_mesh = current_mesh.simplify_quadric_decimation(step_target)
            
            current_faces = len(current_mesh.faces)
            step_target = max(target_faces, current_faces // 2)
            
        except Exception as e:
            print(f"[Decimation] Step simplification failed at {step_target}: {e}")
            # 如果失败，返回当前最佳结果
            break
    
    print(f"[Decimation] Final face count: {len(current_mesh.faces)}")
    return current_mesh


def auto_orient_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    自动对齐：利用主惯性轴（principal inertia axes）把物体对齐世界坐标。
    这样导出的 mesh 在不同物体之间有一致的姿态，便于做比较/展示。
    """
    mesh = mesh.copy()
    # principal_inertia_transform: 4x4 矩阵，把网格变换到主惯性坐标系
    T = mesh.principal_inertia_transform
    mesh.apply_transform(np.linalg.inv(T))
    return mesh


def normalize_bounding_box(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    把 mesh 归一化到 [-0.5, 0.5]^3 的单位立方体，并让模型居中。
    步骤：
      1) 平移到质心在原点
      2) 根据最大边长做缩放，让 max(extents)=1
    """
    mesh = mesh.copy()
    # 平移到中心
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)

    # extents: 轴对齐包围盒的边长
    extents = mesh.bounding_box.extents
    max_extent = float(extents.max())
    if max_extent <= 0:
        print("[Normalize] max extent <= 0, skip scaling.")
        return mesh

    scale = 1.0 / max_extent
    S = np.eye(4)
    S[:3, :3] *= scale
    mesh.apply_transform(S)

    return mesh


def process_mesh(input_path: str,
                 output_path: str,
                 target_faces: int = 5000,
                 smooth_iter: int = 5,
                 smooth_lambda: float = 0.5) -> None:
    """主 pipeline：依次执行所有处理步骤并保存结果。"""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    print(f"Loading mesh from: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        # 某些 OBJ 可能是 Scene，需要合并成一个 mesh
        print("Input is a Scene, merging into a single mesh...")
        mesh = trimesh.util.concatenate(mesh.dump())

    print_mesh_info("Raw mesh", mesh)

    # 对于大型网格，先简化再处理，减少内存压力
    current_faces = len(mesh.faces)
    if current_faces > 200000 and target_faces < current_faces:
        print("Large mesh detected, simplifying first to reduce memory usage...")
        intermediate_target = min(target_faces * 3, current_faces // 2)
        mesh = decimate_mesh_safe(mesh, target_faces=intermediate_target)
        print_mesh_info("After initial simplification", mesh)

    # 1) 平滑
    mesh = laplacian_smoothing(mesh, iterations=smooth_iter, lamb=smooth_lambda)
    print_mesh_info("After smoothing", mesh)

    # 2) 填洞
    mesh = fill_mesh_holes(mesh)
    print_mesh_info("After hole filling", mesh)

    # 3) 最终简化到目标面数
    mesh = decimate_mesh_safe(mesh, target_faces=target_faces)
    print_mesh_info("After decimation", mesh)

    # 4) 自动对齐
    mesh = auto_orient_mesh(mesh)
    print_mesh_info("After auto orientation", mesh)

    # 5) 归一化
    mesh = normalize_bounding_box(mesh)
    print_mesh_info("Final normalized mesh", mesh)

    # 保存
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(output_path)
    print(f"\n[Done] Processed mesh saved to:\n  {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module B: Mesh post-processing for NeRF / instant-ngp exports"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input mesh (OBJ/PLY) exported from instant-ngp",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save processed mesh (OBJ)",
    )
    parser.add_argument(
        "--target_faces",
        type=int,
        default=5000,
        help="Target number of faces after decimation (default: 5000)",
    )
    parser.add_argument(
        "--smooth_iter",
        type=int,
        default=5,
        help="Number of Laplacian smoothing iterations (default: 5)",
    )
    parser.add_argument(
        "--smooth_lambda",
        type=float,
        default=0.5,
        help="Laplacian smoothing step size lambda (default: 0.5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_mesh(
        input_path=args.input,
        output_path=args.output,
        target_faces=args.target_faces,
        smooth_iter=args.smooth_iter,
        smooth_lambda=args.smooth_lambda,
    )