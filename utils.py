import numpy as np
import re
import open3d as o3d
from skimage import measure
from scipy.ndimage import gaussian_filter
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


def get_mesh_params(
    points: list[np.ndarray], grid_size: int = 256
) -> (np.ndarray, np.ndarray, float, (np.ndarray, np.ndarray)):
    """
    Compute mesh grid parameters from a set of 3D points.

    Parameters
    ----------
    points : np.ndarray or list of np.ndarray
        Input 3D points as a single array or a list of arrays.
    grid_size : int, optional
        Number of grid divisions along the largest axis (default is 256).

    Returns
    -------
    origin : np.ndarray
        The minimum coordinates of the bounding box.
    scale : np.ndarray
        The size of the bounding box along each axis.
    voxel_size : float
        The size of each voxel in the grid.
    (min_coords, max_coords) : tuple of np.ndarray
        The minimum and maximum coordinates of the bounding box.
    """
    if isinstance(points, list):
        all_points = np.vstack(points)
    else:
        all_points = points.copy()

    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    origin = min_coords.copy()
    scale = max_coords - min_coords
    voxel_size = scale.max() / grid_size
    return origin, scale, voxel_size, (min_coords, max_coords)


def build_voxel_grid(
    points: list[np.ndarray], grid_size: int = 256
) -> (np.ndarray, np.ndarray, float):
    """
    Constructs a voxel grid from one or more arrays of 3D points.

    Parameters
    ----------
    points : np.ndarray or list of np.ndarray
        Input 3D points as a single array or a list of arrays.
    grid_size : int, optional
        Number of grid divisions along each axis (default is 256).

    Returns
    -------
    voxel_grid : np.ndarray
        3D array of shape (grid_size, grid_size, grid_size) with 0s and 1s indicating occupied voxels.
    origin : np.ndarray
        Minimum coordinates of all points (grid origin).
    voxel_size : float
        Size of a single voxel.
    """
    if isinstance(points, list):
        all_points = np.vstack(points)
    else:
        all_points = points.copy()

    origin, scale, voxel_size, (min_coords, max_coords) = get_mesh_params(
        all_points, grid_size
    )

    # Нормализация координат в диапазон [0, GRID_SIZE]
    normalized = (all_points - min_coords) / voxel_size

    # Преобразуем в сетку
    voxel_indices = np.floor(normalized).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)

    # Заполняем воксельную структуру
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

    return voxel_grid, origin, voxel_size


def mesh_from_voxels(
    voxel_grid: np.ndarray,
    voxel_size: float = 1.0,
    origin: np.ndarray = np.array([0, 0, 0]),
    level: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """
    Converts a voxel grid to a mesh using the marching cubes algorithm.

    Args:
        voxel_grid (np.ndarray): 3D array of 0s and 1s representing filled voxels.
        voxel_size (float, optional): Size of each voxel. Defaults to 1.0.
        origin (np.ndarray, optional): Origin point in camera coordinates. Defaults to [0, 0, 0].
        level (float, optional): Isosurface value for marching cubes. Defaults to 0.5.

    Returns:
        o3d.geometry.TriangleMesh: Generated mesh from the voxel grid.
    """

    vertices, faces, normals, values = measure.marching_cubes(
        voxel_grid, level=level, spacing=(voxel_size, voxel_size, voxel_size)
    )

    # Переводим обратно в координаты воксельной сетки
    cam_vertices = vertices * voxel_size + origin

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(cam_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh


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


def pairwise_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance_coarse: float,
    max_correspondence_distance_fine: float,
) -> (np.ndarray, np.ndarray):
    """
    Performs pairwise registration between two point clouds using a two-stage ICP process.

    First applies coarse ICP with point-to-point estimation, then refines with fine ICP using point-to-plane estimation.
    Returns the final transformation matrix and the information matrix.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance_coarse (float): Maximum correspondence distance for coarse ICP.
        max_correspondence_distance_fine (float): Maximum correspondence distance for fine ICP.

    Returns:
        tuple: (transformation_icp (numpy.ndarray), information_icp (numpy.ndarray))
    """
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        np.identity(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    transformation_icp = icp_fine.transformation

    information_icp = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine, icp_fine.transformation
        )
    )

    return transformation_icp, information_icp


def full_registration(
    pcds: list[o3d.geometry.PointCloud],
    max_correspondence_distance_coarse: float,
    max_correspondence_distance_fine: float,
) -> o3d.pipelines.registration.PoseGraph:
    """
    Performs full multiway registration of a list of point clouds using pairwise ICP and constructs a pose graph.

    Args:
        pcds (list of o3d.geometry.PointCloud): List of point clouds to register.
        max_correspondence_distance_coarse (float): Maximum correspondence distance for coarse ICP.
        max_correspondence_distance_fine (float): Maximum correspondence distance for fine ICP.

    Returns:
        o3d.pipelines.registration.PoseGraph: The constructed pose graph with nodes and edges representing the registration results.
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                source=pcds[source_id],
                target=pcds[target_id],
                max_correspondence_distance_coarse=max_correspondence_distance_coarse,
                max_correspondence_distance_fine=max_correspondence_distance_fine,
            )
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


def merge_meshes(
    meshes: list,
    grid_size: int,
    mcd_coarse_scale: int = 30,
    mcd_fine_scale: int = 7,
    down_sample: bool = False,
) -> (o3d.geometry.PointCloud, float):
    """
    Merges multiple meshes by converting them to point clouds, estimating normals, and performing multiway registration and global optimization to align and combine them into a single point cloud.

    Args:
        meshes (list): List of input meshes as arrays of vertices.
        grid_size (int): Number of grid divisions for voxel size calculation.
        mcd_coarse_scale (int, optional): Scale factor for coarse correspondence distance. Defaults to 30.
        mcd_fine_scale (int, optional): Scale factor for fine correspondence distance. Defaults to 7.
        down_sample (bool, optional): Whether to downsample the merged point cloud. Defaults to False.

    Returns:
        tuple: (o3d.geometry.PointCloud, float) — The merged point cloud with normals and the voxel size used.
    """
    # Считаем размер одного вокселя
    _, _, voxel_size, _ = get_mesh_params(meshes, grid_size)

    # Задаём параметры для объединения мешей
    max_correspondence_distance_coarse = voxel_size * mcd_coarse_scale
    max_correspondence_distance_fine = voxel_size * mcd_fine_scale

    # Конфертируем все меши в PointCloud
    pcds_raw = [vertices_to_pcd(m) for m in meshes]
    # Считаем нормали для PointCloud
    pcds_down = [get_normals_to_pcd(pcd) for pcd in pcds_raw]

    # Посмотрим как взаиморасположены меши
    o3d.visualization.draw_geometries(pcds_down)

    # Находим пересечения PointCloud
    pose_graph = full_registration(
        pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine
    )

    # Подбираем оптимальные параметры транформации для объединения
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # Трансформируем все PointCloud
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

    # Объединяем точки в один PointCloud
    pcds_raw = [vertices_to_pcd(m) for m in meshes]
    pcds = [get_normals_to_pcd(pcd) for pcd in pcds_raw]
    pcd_combined = o3d.pybind.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    res_psd = pcd_combined
    if down_sample:
        res_psd = pcd_combined.voxel_down_sample(voxel_size=float(voxel_size))

    # Пересчитаем нормали объединённого PointCloud
    pcd_norm = get_normals_to_pcd(res_psd)
    return pcd_norm, voxel_size


def pcd_to_voxel_grid(
    points: np.ndarray, grid_size: int = 256, apply_filter: bool = False
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Converts a point cloud to a voxel grid representation.

    Normalizes the input points to fit within a cubic grid of specified size,
    fills the grid based on point occupancy, and optionally applies a Gaussian filter.

    Args:
        points (np.ndarray): Input point cloud of shape (N, 3).
        grid_size (int, optional): Size of the voxel grid along each axis. Defaults to 256.
        apply_filter (bool, optional): Whether to apply a Gaussian filter to the voxel grid. Defaults to False.

    Returns:
        tuple: (volume, min_bounds, scale) where
            volume (np.ndarray): The resulting voxel grid.
            min_bounds (np.ndarray): Minimum bounds of the original points.
            scale (np.ndarray): Scale used for normalization.
    """
    # Нормализуем точки в диапазон [0, grid_size)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    scale = max_bounds - min_bounds
    normalized = (points - min_bounds) / scale
    indices = (normalized * (grid_size - 1)).astype(np.int16)

    # Создаем пустую воксельную сетку
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.int8)

    # Заполняем её (можно аккумулировать плотность или просто наличие точек)
    for idx in indices:
        volume[tuple(idx)] = 1

    if apply_filter:
        volume = gaussian_filter(volume, sigma=0.5)

    return volume, min_bounds, scale


def marching_cubes(
    voxel_grid: np.ndarray, voxel_size: float, level: float = 0.5
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Extracts a mesh from a 3D voxel grid using the marching cubes algorithm.

    Args:
        voxel_grid (np.ndarray): 3D array representing the voxel grid.
        voxel_size (float): Size of each voxel.
        level (float, optional): Isosurface value to extract. Defaults to 0.5.

    Returns:
        tuple: vertices (np.ndarray), faces (np.ndarray), normals (np.ndarray), values (np.ndarray) of the extracted mesh.
    """
    vertices, faces, normals, values = measure.marching_cubes(
        voxel_grid, level=level, spacing=(voxel_size, voxel_size, voxel_size)
    )
    return vertices, faces, normals, values
