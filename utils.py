import numpy as np
import re
import open3d as o3d
from tqdm import tqdm


def parse_x_file(filepath: str) -> dict:
    """
    Parses a .x file to extract mesh data, including vertices, faces, UV coordinates, and transformation matrices.

    Args:
        filepath (str): Path to the .x file.

    Returns:
        dict: Dictionary containing 'vertices' (np.ndarray), 'faces' (np.ndarray), 'uvs' (np.ndarray),
              'rv_matrix' (np.ndarray), and 'frame_matrix' (np.ndarray).

    Raises:
        ValueError: If required sections or data are missing in the file.
    """
    with open(filepath, "r") as f:
        text = f.read()

    # --- Извлечение RV_Calibration (матрица перехода из мировой системы координат в систему координат камеры) ---
    rv_match = re.search(r"RV_Calibration\s*{\s*((?:-?\d+\.\d+[,;]\s*){16})", text)
    if not rv_match:
        raise ValueError("RV_Calibration не найдена")

    matrix_values = list(map(float, re.findall(r"-?\d+\.\d+", rv_match.group(1))))
    rv_matrix = np.array(matrix_values).reshape((4, 4))

    # Извлечение FrameTransformMatrix (матрица перехода из локальных координат в мировые)
    frame_match = re.search(
        r"FrameTransformMatrix\s*{\s*((?:-?\d+\.\d+[,;]\s*){16})", text, re.DOTALL
    )
    if not frame_match:
        raise ValueError("FrameTransformMatrix не найдена")
    frame_values = list(map(float, re.findall(r"-?\d+\.\d+", frame_match.group(1))))
    frame_matrix = np.array(frame_values).reshape((4, 4))

    # --- Извлечение количества вершин ---
    mesh_header = re.search(r"Mesh\s+\w+\s*{\s*(\d+);", text)
    if not mesh_header:
        raise ValueError("Mesh-блок не найден")
    num_vertices = int(mesh_header.group(1))
    print(f"num_vertices: {num_vertices}")

    # --- Извлечение самих вершин (Nx3) ---
    vertex_pattern = r"(-?\d+\.\d+);(-?\d+\.\d+);(-?\d+\.\d+);[;,]"
    vertex_data = re.findall(vertex_pattern, text)
    if len(vertex_data) < num_vertices:
        raise ValueError("Недостаточно вершин найдено")
    vertices = np.array(vertex_data[:num_vertices], dtype=np.float32)

    # --- Извлечение количества треугольников ---
    face_start_idx = text.find(vertex_data[num_vertices - 1][2])
    face_block_match = re.search(
        r"(\d+);\s*(3;[^\}]+?);", text[face_start_idx:], re.DOTALL
    )
    if not face_block_match:
        raise ValueError("Блок треугольников не найден")
    num_faces = int(face_block_match.group(1))
    print(f"num_faces: {num_faces}")

    # --- Извлечение индексов треугольников ---
    face_pattern = r"3;(\d+),(\d+),(\d+);[;,]"
    face_data = re.findall(face_pattern, text)
    if len(face_data) < num_faces:
        raise ValueError("Недостаточно треугольников найдено")
    faces = np.array(face_data[:num_faces], dtype=np.int32)

    # --- Извлечение UV-координат ---
    uv_match = re.search(r"MeshTextureCoords\s*{\s*(\d+);", text)
    if not uv_match:
        raise ValueError("UV-координаты не найдены")
    num_uvs = int(uv_match.group(1))
    print(f"num_uvs: {num_uvs}")
    uv_data = re.findall(r"(-?\d+\.\d+);(-?\d+\.\d+);[;,]", text[uv_match.start() :])
    if len(uv_data) < num_uvs:
        raise ValueError("Недостаточно UV-координат")
    uvs = np.array(uv_data[:num_uvs], dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "rv_matrix": rv_matrix,
        "frame_matrix": frame_matrix,
    }


def apply_transformation(vertices: np.ndarray, matrix4x4: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to a set of 3D vertices.

    Parameters
    ----------
    vertices : np.ndarray
        An (N, 3) array of 3D vertex coordinates.
    matrix4x4 : np.ndarray
        A (4, 4) transformation matrix.

    Returns
    -------
    np.ndarray
        An (N, 3) array of transformed 3D vertices.
    """
    # (x, y, z) -> (x, y, z, 1)
    ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
    vertices_hom = np.hstack([vertices, ones])  # (N, 4)

    # Применим калибровку
    transformed = vertices_hom @ matrix4x4  # (N, 4)
    return transformed[:, :3]


def vertices_to_pcd(vertices: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converts a NumPy array of vertices to an Open3D PointCloud object.

    Args:
        vertices (np.ndarray): Array of 3D points with shape (N, 3).

    Returns:
        o3d.geometry.PointCloud: PointCloud containing the input vertices.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    return pcd


def get_normals_to_pcd(
    pcd: o3d.geometry.PointCloud, consistent: bool = False, radius=5.0, max_nn=30
) -> o3d.geometry.PointCloud:
    """
    Estimates and optionally orients normals for a given point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        consistent (bool, optional): If True, orients normals consistently using tangent plane. Defaults to False.
        radius (float, optional): Search radius for normal estimation. Defaults to 5.0.
        max_nn (int, optional): Maximum nearest neighbors for normal estimation. Defaults to 30.

    Returns:
        o3d.geometry.PointCloud: Point cloud with estimated (and possibly oriented) normals.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    if consistent:
        pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd


def compute_triangle_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculates the normal vector of a triangle defined by three 3D points.

    Args:
        p0 (np.ndarray): First vertex of the triangle.
        p1 (np.ndarray): Second vertex of the triangle.
        p2 (np.ndarray): Third vertex of the triangle.

    Returns:
        np.ndarray: The normal vector of the triangle.
    """
    return np.cross(p1 - p0, p2 - p0)


def point_to_triangle_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Calculates the shortest distance from a point to a triangle in 3D space.

    Parameters
    ----------
    p : np.ndarray
        The 3D coordinates of the point.
    a : np.ndarray
        The 3D coordinates of the first triangle vertex.
    b : np.ndarray
        The 3D coordinates of the second triangle vertex.
    c : np.ndarray
        The 3D coordinates of the third triangle vertex.

    Returns
    -------
    float
        The minimum distance from the point to the triangle.
    """
    # Истинное расстояние от точки до треугольника в 3D
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        # точка p лежит вне треугольника со стороны вершины a
        return np.linalg.norm(ap)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        # точка p лежит вне треугольника со стороны вершины b
        return np.linalg.norm(bp)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        # точка p ближе всего к ребру AB.
        return np.linalg.norm(p - proj)

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        # точка p лежит вне треугольника со стороны вершины c
        return np.linalg.norm(cp)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        # точка p ближе всего к ребру AC
        return np.linalg.norm(p - proj)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        # точка p ближе всего к ребру BC.
        return np.linalg.norm(p - proj)

    # точка p лежит внутри треугольника
    n = np.cross(ab, ac)
    n /= np.linalg.norm(n)
    return abs(np.dot(ap, n))


def compute_weight(p: np.ndarray, camera_center: np.ndarray, normal: np.ndarray):
    """
    Calculates the weight based on the angle between the viewing direction and the surface normal.

    Parameters
    ----------
    p : np.ndarray
        The 3D point of interest.
    camera_center : np.ndarray
        The position of the camera in 3D space.
    normal : np.ndarray
        The normal vector at point p.

    Returns
    -------
    float
        The computed weight as the absolute value of the dot product between the normalized viewing direction and the normal.
    """
    direction = camera_center - p
    direction /= np.linalg.norm(direction)
    weight = abs(np.dot(direction, normal))
    return weight


def integrate_mesh_to_voxel_grid(mesh: dict, voxel_space: dict) -> None:
    """
    Integrates a 3D mesh into a voxel grid by updating the distance and weight volumes for each voxel intersecting the mesh triangles.

    Parameters
    ----------
    mesh : dict
        Dictionary containing mesh data with keys "vertices", "faces", and "frame_matrix".
    voxel_space : dict
        Dictionary containing voxel grid data with keys "D", "W", "min_bound", and "voxel_size".

    Returns
    -------
    None
        Updates the voxel grid in place.
    """
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    frame_matrix = mesh["frame_matrix"]
    camera_center = apply_transformation(np.array([[0.0, 0.0, 0.0]]), frame_matrix)[0]

    D = voxel_space["D"]
    W = voxel_space["W"]
    min_bound = voxel_space["min_bound"]
    voxel_size = voxel_space["voxel_size"]

    for face in tqdm(faces, desc="Faces", unit=" triangles", unit_scale=1):
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = compute_triangle_normal(a, b, c)
        normal /= np.linalg.norm(normal)

        # Bounding box треугольника в МИРОВЫХ координата
        tri_min = np.minimum(np.minimum(a, b), c)
        tri_max = np.maximum(np.maximum(a, b), c)

        # Переводим его в индексы воксельной сетки
        min_idx = np.floor((tri_min - min_bound) / voxel_size).astype(int)
        max_idx = np.ceil((tri_max - min_bound) / voxel_size).astype(int)

        for i in range(min_idx[0], max_idx[0] + 1):
            for j in range(min_idx[1], max_idx[1] + 1):
                for k in range(min_idx[2], max_idx[2] + 1):
                    voxel_center = min_bound + voxel_size * (np.array([i, j, k]) + 0.5)

                    distance = point_to_triangle_distance(voxel_center, a, b, c)
                    weight = compute_weight(voxel_center, camera_center, normal)

                    if weight == 0.0:
                        continue  # скипаем неинтересные места

                    D[i, j, k] = (W[i, j, k] * D[i, j, k] + weight * distance) / (
                        W[i, j, k] + weight
                    )
                    W[i, j, k] += weight


def save_mesh_to_x_format(
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    texture_coords: np.ndarray | None,
    texture_filename: str | None,
) -> None:
    """
    Saves a mesh to the .x (DirectX text) format.

    Parameters
    ----------
    filename : str
        Path to the output .x file.
    vertices : np.ndarray
        Mesh vertices of shape (N, 3).
    faces : np.ndarray
        Mesh triangles of shape (M, 3).
    normals : np.ndarray or None
        Optional vertex normals of shape (N, 3).
    texture_coords : np.ndarray or None
        Optional texture coordinates of shape (N, 2).
    texture_filename : str or None
        Optional texture filename to reference in the .x file.

    Returns
    -------
    None
    """

    with open(filename, "w") as f:
        f.write("xof 0302txt 0032\n")
        f.write("Header {\n1;\n0;\n1;\n}\n")
        f.write("Mesh Scene {\n")

        # Точки
        f.write(f"{len(vertices)};\n")
        for v in tqdm(vertices, desc="Vertices", unit=" points", unit_scale=1):
            f.write(f"{v[2]};{v[1]};{v[0]};,\n")
        f.write("\n")

        # Треугольники
        f.write(f"{len(faces)};\n")
        for face in tqdm(faces, desc="Faces", unit=" triangles", unit_scale=1):
            f.write(f"3;{face[2]},{face[1]},{face[0]};,\n")
        f.write("\n")

        # Нормали
        if normals is not None and len(normals) == len(vertices):
            f.write("MeshNormals {\n")
            f.write(f"{len(normals)};\n")
            for n in tqdm(
                normals, desc="Normals Vertices", unit=" vectors", unit_scale=1
            ):
                f.write(f"{n[2]};{n[1]};{n[0]};,\n")
            f.write("\n")
            f.write(f"{len(faces)};\n")
            for face in tqdm(
                normals, desc="Normals Faces", unit=" vectors", unit_scale=1
            ):
                f.write(f"3;{face[2]},{face[1]},{face[0]};,\n")
            f.write("}\n\n")

        # Координаты тектстуры
        if texture_coords is not None and len(texture_coords) == len(vertices):
            if texture_filename:
                f.write("TextureFilename {\n")
                f.write(f'"{texture_filename}"\n')
                f.write("}\n")
            f.write("MeshTextureCoords {\n")
            f.write(f"{len(texture_coords)};\n")
            for t in tqdm(
                texture_coords, desc="Texture coords", unit=" coords", unit_scale=1
            ):
                f.write(f"{t[0]};{t[1]};,\n")
            f.write("}\n")

        f.write("}\n")


def write_mesh_to_x(
    mesh: o3d.geometry.TriangleMesh,
    filename: str,
    save_normals: bool = True,
    save_texture: bool = True,
) -> None:
    """
    Exports an Open3D TriangleMesh to the .x (DirectX text) format.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The mesh to export.
    filename : str
        Path to the output .x file.
    save_normals : bool, optional
        Whether to include vertex normals in the export. Default is True.
    save_texture : bool, optional
        Whether to include texture coordinates and texture filename in the export. Default is True.

    Returns
    -------
    None
    """

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = texture_coords = texture_filename = None
    if save_normals:
        normals = np.asarray(mesh.vertex_normals)

    if save_texture:
        texture_coords = np.asarray(mesh.texture_coords)
        texture_filename = str(mesh.texture_filename)

    save_mesh_to_x_format(
        filename=filename,
        vertices=vertices,
        faces=faces,
        normals=normals,
        texture_coords=texture_coords,
        texture_filename=texture_filename,
    )
