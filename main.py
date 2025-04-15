import numpy as np
import re
import open3d as o3d
import os

def parse_x_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    # --- Извлечение RV_Calibration (матрица перехода из мировой системы координат в систему координат камеры) ---
    rv_match = re.search(r'RV_Calibration\s*{\s*((?:-?\d+\.\d+[,;]\s*){16})', text)
    if not rv_match:
        raise ValueError("RV_Calibration не найдена")

    matrix_values = list(map(float, re.findall(r'-?\d+\.\d+', rv_match.group(1))))
    rv_matrix = np.array(matrix_values).reshape((4, 4))

    # --- Извлечение количества вершин ---
    mesh_header = re.search(r'Mesh\s+\w+\s*{\s*(\d+);', text)
    if not mesh_header:
        raise ValueError("Mesh-блок не найден")
    num_vertices = int(mesh_header.group(1))
    print(f"num_vertices: {num_vertices}")

    # --- Извлечение самих вершин (Nx3) ---
    vertex_pattern = r'(-?\d+\.\d+);(-?\d+\.\d+);(-?\d+\.\d+);[;,]'
    vertex_data = re.findall(vertex_pattern, text)
    print(f"vertex_data[0]: {vertex_data[0]}")
    print(f"vertex_data[-1]: {vertex_data[-1]}")
    print(f"len(vertex_data): {len(vertex_data)}")
    if len(vertex_data) < num_vertices:
        raise ValueError("Недостаточно вершин найдено")
    vertices = np.array(vertex_data[:num_vertices], dtype=np.float32)

    # --- Извлечение количества треугольников ---
    face_start_idx = text.find(vertex_data[num_vertices - 1][2])  # позиция последней вершины
    face_block_match = re.search(r'(\d+);\s*(3;[^\}]+?);', text[face_start_idx:], re.DOTALL)
    if not face_block_match:
        raise ValueError("Блок треугольников не найден")
    num_faces = int(face_block_match.group(1))
    print(f"num_faces: {num_faces}")

    # --- Извлечение индексов треугольников ---
    face_pattern = r'3;(\d+),(\d+),(\d+);[;,]'
    face_data = re.findall(face_pattern, text)
    print(f"face_data[0]: {face_data[0]}")
    print(f"face_data[-1]: {face_data[-1]}")
    print(f"len(face_data): {len(face_data)}")
    if len(face_data) < num_faces:
        raise ValueError("Недостаточно треугольников найдено")
    faces = np.array(face_data[:num_faces], dtype=np.int32)

    # --- Извлечение UV-координат ---
    uv_match = re.search(r'MeshTextureCoords\s*{\s*(\d+);', text)
    uvs = None
    if uv_match:
        num_uvs = int(uv_match.group(1))
        print(f"num_uvs: {num_uvs}")
        uv_data = re.findall(r'(-?\d+\.\d+);(-?\d+\.\d+);[;,]', text[uv_match.start():])
        print(f"uv_data[0]: {uv_data[0]}")
        print(f"uv_data[-1]: {uv_data[-1]}")
        print(f"len(uv_data): {len(uv_data)}")
        if len(uv_data) >= num_uvs:
            uvs = np.array(uv_data[:num_uvs], dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "rv_matrix": rv_matrix
    }


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(ROOT_DIR, 'data\\teapot_1.x')


res = parse_x_file(filepath)

