import os
import numpy as np
import polyscope as ps
import nimblephysics as nimble

# Additional imports for mesh and XML parsing
import pyvista as pv
import xml.etree.ElementTree as ET


# =============================================================
# Configure your motion file here (same style as visualize_frames.py)
# =============================================================

# Baseline Comparisons
# 1
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1479.b3d" # Back bend: Reference
# motion_file = "/home/ncking/git_repositories/diagrams/motions/50.b3d" # Back bend: Baseline
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1479-back_bend.npy" # Back bend: PG-BIG

# 15
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1479.b3d" # Lunge: Reference
# motion_file = "/home/ncking/git_repositories/diagrams/motions/457.b3d" # Lunge: Baseline
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1479-lunge_left.npy" # Lunge: PG-BIG

# 28
# motion_file = "/home/ncking/git_repositories/diagrams/motions/50.b3d" # T-Balance: Reference
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1479.b3d" # T-Balance: Baseline
# motion_file = "/home/ncking/git_repositories/diagrams/motions/50-t_balance_right.npy" # T-Balance: PG-BIG

# 19
# motion_file = "/home/ncking/git_repositories/diagrams/motions/50.b3d" # T-Balance: Reference
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1479.b3d" # T-Balance: Baseline
# motion_file = "/home/ncking/git_repositories/diagrams/motions/50-scorpion.npy" # T-Balance: PG-BIG

# 6
# motion_file = "/home/ncking/git_repositories/diagrams/motions/46.b3d" # T-Balance: Reference
# motion_file = "/home/ncking/git_repositories/diagrams/motions/921.b3d" # T-Balance: Baseline
# motion_file = "/home/ncking/git_repositories/diagrams/motions/761.b3d" # T-Balance: Baseline

# Age
# motion_file = "/home/ncking/git_repositories/diagrams/motions/46-drop_jump.npy"
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1127-t_balance_left.npy"
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1160-t_balance_left.npy"


# motion_file = "/home/ncking/git_repositories/diagrams/motions/754.b3d"
# motion_file = "/home/ncking/git_repositories/diagrams/motions/38.b3d"
# motion_file = "/home/ncking/git_repositories/diagrams/motions/1145.b3d"

motion_file = "/home/ncking/git_repositories/diagrams/motions/46.b3d"


trial_idx = 24 # Only used for .b3d files


# =============================================================
# Visualization configuration (parity with visualize_frames.py)
# =============================================================
num_poses = 10      # Number of poses to display
spacing = 0.5     # Spacing between skeletons along X

show_floor = True   # Polyscope ground plane

# Trajectory visualization
show_trajectory = True
trajectory_body_node = "humerus_l"  # e.g., "pelvis", "calcn_l", "humerus_l"
trajectory_line_radius = 0.005   # thickness for trajectory curve

# Motion breakpoints (normalized [0,1])
motion_start = 0 # 0.0 = first frame
motion_end = 1.0   # 1.0 = last frame

# Transparency gradient (alpha only)
use_color_gradient = False
use_alpha_gradient = False
start_alpha = 0.5
end_alpha = 0.7
default_color = [1, 1, 1]

# Optional: render a support box under the first pose
show_support_box = False      # set True to show the box
support_box_height = 0.30     # meters (30 cm)
support_box_length = 0.40     # X dimension (meters)
support_box_width = 0.40      # Z dimension (meters)
support_box_ground_height = 0.0  # y=0 ground plane height


def frames_from_poses(arr, target_dofs=37):
    arr = np.asarray(arr)

    if arr.ndim == 1:
        if arr.size == target_dofs:
            return arr[np.newaxis, :]
        if arr.size < target_dofs:
            pad = np.zeros(target_dofs - arr.size)
            return np.hstack([arr, pad])[np.newaxis, :]
        return arr[:target_dofs][np.newaxis, :]

    if arr.ndim == 2:
        if arr.shape[1] == target_dofs:
            return arr.copy()
        if arr.shape[0] == target_dofs:
            return arr.T.copy()
        if arr.shape[1] < target_dofs:
            pad = np.zeros((arr.shape[0], target_dofs - arr.shape[1]))
            return np.hstack([arr, pad])
        return arr[:, :target_dofs]

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim == 2:
            if arr.shape[0] == target_dofs:
                return arr.T.copy()
            if arr.shape[1] == target_dofs:
                return arr.copy()
            if arr.shape[1] < target_dofs:
                pad = np.zeros((arr.shape[0], target_dofs - arr.shape[1]))
                return np.hstack([arr, pad])
            return arr[:, :target_dofs]

        axes = arr.shape
        if target_dofs in axes:
            chan_axis = int(np.where(np.array(axes) == target_dofs)[0][0])
            moved = np.moveaxis(arr, chan_axis, -1)
            frames = moved.reshape(-1, target_dofs)
            return frames

        frames = arr.reshape(-1, arr.shape[-1])
        if frames.shape[1] < target_dofs:
            pad = np.zeros((frames.shape[0], target_dofs - frames.shape[1]))
            return np.hstack([frames, pad])
        return frames[:, :target_dofs]

    raise ValueError(f"Unsupported poses ndim: {arr.ndim}")


def load_motion(motion_path, trial_index=0):
    file_ext = os.path.splitext(motion_path)[1].lower()
    if file_ext == ".b3d":
        subject = nimble.biomechanics.SubjectOnDisk(motion_path)
        trial_length = subject.getTrialLength(trial_index)
        frames = subject.readFrames(
            trial=trial_index,
            startFrame=0,
            numFramesToRead=trial_length,
            includeSensorData=False,
            includeProcessingPasses=True,
        )
        kin_passes = [frame.processingPasses[0] for frame in frames if frame.processingPasses]
        poses = np.array([kp.pos for kp in kin_passes if hasattr(kp, "pos")])
        timestep = subject.getTrialTimestep(trial_index)
        return poses, timestep
    elif file_ext == ".npy":
        poses = np.load(motion_path)
        timestep = 1.0 / 120.0
        return poses, timestep
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please use .b3d or .npy files.")


def skeleton_world_points(skeleton: nimble.dynamics.Skeleton):
    """Return body node world positions (N x 3) and parent-child edges."""
    names = []
    positions = []
    for i in range(skeleton.getNumBodyNodes()):
        bn = skeleton.getBodyNode(i)
        tf = bn.getWorldTransform()
        positions.append(np.array(tf.translation(), dtype=float))
        names.append(bn.getName())
    positions = np.asarray(positions)

    # Build edges: connect each body to its parent body if exists
    name_to_index = {n: i for i, n in enumerate(names)}
    edges = []
    for i in range(skeleton.getNumBodyNodes()):
        bn = skeleton.getBodyNode(i)
        parent = bn.getParentBodyNode()
        if parent is not None:
            a = name_to_index[parent.getName()]
            b = i
            edges.append([a, b])
    edges = np.array(edges, dtype=int) if len(edges) > 0 else np.zeros((0, 2), dtype=int)
    return positions, edges


def get_body_mesh_map_auto(geometry_folder, body_names):
    """
    Map every mesh file to the closest matching body node in the Rajagopal model.
    Returns dict mapping body_name -> list of mesh paths (can be multiple meshes per body).
    """
    if not os.path.isdir(geometry_folder):
        return {}
    mesh_files = [f for f in os.listdir(geometry_folder) if f.lower().endswith('.vtp')]
    mesh_basenames = [os.path.splitext(f)[0].lower() for f in mesh_files]

    # Explicit anatomical mesh-to-body-node mappings
    anatomical_mesh_map = {
        'sacrum': 'sacrum',
        'hat_skull': 'head',  # Change 'head' to your skeleton's head node name if needed
        'hat_jaw': 'head',    # Change 'head' to your skeleton's head node name if needed
        'hat_ribs_scap': 'torso',
    }

    special_mappings = {
        'calcn_r': 'r_foot',
        'calcn_l': 'l_foot',
        'toes_r': 'r_bofoot',
        'toes_l': 'l_bofoot',
        'torso': 'hat_spine',
        'humerus_r': 'humerus_rv',
        'humerus_l': 'humerus_lv',
        'ulna_r': 'ulna_rv',
        'ulna_l': 'ulna_lv',
        'radius_r': 'radius_rv',
        'radius_l': 'radius_lv',
    }

    mesh_map = {bn: [] for bn in body_names}
    # First pass: use existing heuristics
    for body_name in body_names:
        name = body_name.lower()
        candidates = []
        # Check special mappings first
        if name in special_mappings:
            alt = special_mappings[name]
            if alt in mesh_basenames:
                candidates.append(alt)

        # Direct match
        if name in mesh_basenames:
            candidates.append(name)
        # Suffix match (e.g., femur_r -> r_femur)
        if name.endswith('_r'):
            alt = 'r_' + name[:-2]
            if alt in mesh_basenames:
                candidates.append(alt)
        if name.endswith('_l'):
            alt = 'l_' + name[:-2]
            if alt in mesh_basenames:
                candidates.append(alt)
        # Prefix match (e.g., r_femur -> femur_r)
        if name.startswith('r_'):
            alt = name[2:] + '_r'
            if alt in mesh_basenames:
                candidates.append(alt)
        if name.startswith('l_'):
            alt = name[2:] + '_l'
            if alt in mesh_basenames:
                candidates.append(alt)
        # Try removing _SOMEINVERTEDFACES
        if name.endswith('_someinvertedfaces'):
            alt = name.replace('_someinvertedfaces', '')
            if alt in mesh_basenames:
                candidates.append(alt)

        # Pelvis: always add both l_pelvis and r_pelvis
        if name == 'pelvis':
            for prefix in ['l_', 'r_']:
                alt = prefix + 'pelvis'
                if alt in mesh_basenames:
                    candidates.append(alt)

        # Hand mapping: add all metacarpals and finger segments for the side
        if 'hand' in name:
            side = 'r' if name.endswith('_r') else 'l' if name.endswith('_l') else None
            if side:
                # Add all metacarpals
                for i in range(1, 6):
                    alt = f'metacarpal{i}_{side}vs'
                    if alt in mesh_basenames:
                        candidates.append(alt)
                # Add all finger bones
                for finger in ['thumb', 'index', 'middle', 'ring', 'little']:
                    for segment in ['proximal', 'medial', 'distal']:
                        alt = f'{finger}_{segment}_{side}vs'
                        if alt in mesh_basenames:
                            candidates.append(alt)

        # Wrist/carpus mapping
        if name in ['wrist', 'carpus'] or 'wrist' in name or 'carpus' in name:
            for carpal in ['scaphoid', 'lunate', 'triquetrum', 'pisiform', 'trapezium', 'trapezoid', 'capitate', 'hamate']:
                for side in ['lvs', 'rvs']:
                    alt = f'{carpal}_{side}'
                    if alt in mesh_basenames:
                        candidates.append(alt)

        # Head mapping
        if name in ['head', 'skull', 'neck', 'spine_upper', 'spine_top']:
            for head_mesh in ['hat_skull', 'hat_jaw']:
                if head_mesh in mesh_basenames:
                    candidates.append(head_mesh)

        # Torso/ribs mapping
        if name in ['torso', 'ribs', 'spine'] or 'torso' in name or 'ribs' in name or 'spine' in name:
            for torso_mesh in ['hat_ribs_scap', 'hat_spine', 'sacrum']:
                if torso_mesh in mesh_basenames:
                    candidates.append(torso_mesh)

        # Prefer left/right if applicable
        chosen = []
        if name.endswith('_l'):
            for c in candidates:
                if c.startswith('l_') or c.endswith('_l') or 'lv' in c or '_lvs' in c:
                    chosen.append(c)
        elif name.endswith('_r'):
            for c in candidates:
                if c.startswith('r_') or c.endswith('_r') or 'rv' in c or '_rvs' in c:
                    chosen.append(c)
        if not chosen:
            chosen = candidates
        unique_candidates = []
        for c in chosen:
            if c in mesh_basenames and c not in unique_candidates:
                unique_candidates.append(c)
        for c in unique_candidates:
            idx = mesh_basenames.index(c)
            mesh_map[body_name].append(os.path.join(geometry_folder, mesh_files[idx]))
    # Second pass: assign any unmapped mesh to the correct anatomical node if possible
    mapped_meshes = set()
    for mesh_list in mesh_map.values():
        mapped_meshes.update(mesh_list)
    for mesh_file, mesh_base in zip(mesh_files, mesh_basenames):
        mesh_path = os.path.join(geometry_folder, mesh_file)
        if mesh_path not in mapped_meshes:
            # Use explicit anatomical mapping if available
            if mesh_base in anatomical_mesh_map:
                node = anatomical_mesh_map[mesh_base]
                if node in mesh_map:
                    mesh_map[node].append(mesh_path)
                else:
                    mesh_map[node] = [mesh_path]
            else:
                # Find the closest body node by substring match
                best_node = None
                best_score = 0
                for bn in body_names:
                    score = sum([part in mesh_base for part in bn.lower().split('_')])
                    if score > best_score:
                        best_score = score
                        best_node = bn
                # If no match, assign to 'torso'
                if not best_node:
                    best_node = 'torso'
                mesh_map[best_node].append(mesh_path)
    return mesh_map


def parse_osim_mesh_mapping(osim_path, geometry_folder):
    """
    Parse the .osim XML file to extract mesh file and local transform for each body node.
    Supports Rajagopal2015.osim structure using <VisibleObject>/<GeometrySet>/<objects>/<DisplayGeometry>.
    Returns: dict of body_name -> list of dicts with keys: mesh_path, translation(3), rotation(3 rad), scale(3)
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(osim_path)
    root = tree.getroot()
    # Rajagopal2015.osim does not use XML namespaces for these tags
    mapping = {}
    for body in root.findall('.//Body'):
        body_name = body.get('name')
        if not body_name:
            continue
        # Find DisplayGeometry entries under this Body
        vis = body.find('VisibleObject')
        if vis is None:
            continue
        geom_set = vis.find('GeometrySet')
        if geom_set is None:
            continue
        objects = geom_set.find('objects')
        if objects is None:
            continue
        for dg in objects.findall('DisplayGeometry'):
            mesh_file_el = dg.find('geometry_file')
            if mesh_file_el is None or not (mesh_file_el.text and mesh_file_el.text.strip()):
                continue
            mesh_file = mesh_file_el.text.strip()
            mesh_path = os.path.join(geometry_folder, os.path.basename(mesh_file))
            # Defaults
            rotation = [0.0, 0.0, 0.0]  # radians, order rX rY rZ
            translation = [0.0, 0.0, 0.0]
            scale = [1.0, 1.0, 1.0]
            # <transform> has 6 numbers: rX rY rZ tx ty tz
            tr_el = dg.find('transform')
            if tr_el is not None and tr_el.text:
                vals = [float(x) for x in tr_el.text.strip().split()]
                if len(vals) >= 3:
                    rotation = vals[:3]
                if len(vals) >= 6:
                    translation = vals[3:6]
            sc_el = dg.find('scale_factors')
            if sc_el is not None and sc_el.text:
                svals = [float(x) for x in sc_el.text.strip().split()]
                if len(svals) == 3:
                    scale = svals
            entry = {
                'mesh_path': mesh_path,
                'translation': translation,
                'rotation': rotation,  # radians
                'scale': scale,
            }
            mapping.setdefault(body_name, []).append(entry)
    return mapping

def _euler_body_fixed_xyz_to_matrix(rx, ry, rz):
    """Build rotation matrix for body-fixed XYZ rotations (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # Rotation matrices about body axes X then Y then Z (intrinsic XYZ)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # For body-fixed XYZ, overall R = Rz @ Ry @ Rx (equivalent to intrinsic rotations about X,Y,Z)
    return Rz @ Ry @ Rx


def main():
    # Load and normalize poses to shape (T, 37)
    poses, timestep = load_motion(motion_file, trial_idx)
    expanded = frames_from_poses(poses, target_dofs=37)
    total_frames = len(expanded)

    # Choose frames between motion_start and motion_end
    start_idx = int(round(motion_start * (total_frames - 1)))
    end_idx = int(round(motion_end * (total_frames - 1)))
    if end_idx < start_idx:
        end_idx = start_idx
    frame_indices = np.linspace(start_idx, end_idx, num_poses, dtype=int)

    # Init Polyscope
    ps.init()
    ps.remove_all_structures()
    ps.set_automatically_compute_scene_extents(True)
    ps.set_navigation_style("free")
    if show_floor:
        ps.set_ground_plane_mode('tile_reflection')
    else:
        ps.set_ground_plane_mode('none')

    # Prepare spacing offsets for displayed frames
    display_offsets = {}

    geometry_folder = "/home/ncking/git_repositories/diagrams/Geometry"
    official_mesh_map = parse_osim_mesh_mapping('/home/ncking/git_repositories/diagrams/Rajagopal2015.osim', geometry_folder)

    # Get body node names from skeleton
    model = nimble.RajagopalHumanBodyModel()
    skel = model.skeleton
    body_names = [skel.getBodyNode(i).getName() for i in range(skel.getNumBodyNodes())]

    # Register each pose as a set of body meshes using official mapping
    for i, frame_idx in enumerate(frame_indices):
        model = nimble.RajagopalHumanBodyModel()
        skel = model.skeleton

        offset_x = (i - (num_poses - 1) / 2.0) * spacing
        display_offsets[int(frame_idx)] = float(offset_x)

        pos = expanded[frame_idx].copy()
        if pos.shape[0] > 3:
            pos[3] += offset_x
        skel.setPositions(pos)

        # Optionally render a support box under the first pose
        if show_support_box and i == 0:
            # Place box relative to ground, not subject: base sits at configured ground height
            # Center in XZ under the pelvis of the first pose
            pelvis_node = skel.getBodyNode("pelvis") if skel.getBodyNode("pelvis") else None
            if pelvis_node is not None:
                p = np.array(pelvis_node.getWorldTransform().translation(), dtype=float)
                center_x = float(p[0])
                center_z = float(p[2])
            else:
                # Fallback: center on skeleton average XZ
                xs, zs = [], []
                for bni in range(skel.getNumBodyNodes()):
                    pt = np.array(skel.getBodyNode(bni).getWorldTransform().translation(), dtype=float)
                    xs.append(pt[0]); zs.append(pt[2])
                center_x = float(np.mean(xs)) if xs else 0.0
                center_z = float(np.mean(zs)) if zs else 0.0

            # Box extends upward from the ground plane
            base_y = float(support_box_ground_height) - 1e-4  # tiny sink to avoid z-fighting with ground
            top_y = base_y + support_box_height
            lx = support_box_length * 0.5
            lz = support_box_width * 0.5
            # 8 vertices
            verts_box = np.array([
                [center_x - lx, base_y, center_z - lz],  # 0 bottom
                [center_x + lx, base_y, center_z - lz],  # 1
                [center_x + lx, base_y, center_z + lz],  # 2
                [center_x - lx, base_y, center_z + lz],  # 3
                [center_x - lx, top_y,  center_z - lz],  # 4 top
                [center_x + lx, top_y,  center_z - lz],  # 5
                [center_x + lx, top_y,  center_z + lz],  # 6
                [center_x - lx, top_y,  center_z + lz],  # 7
            ], dtype=float)
            # Triangular faces (12)
            faces_box = np.array([
                [0,1,2], [0,2,3],       # bottom
                [4,5,6], [4,6,7],       # top
                [0,1,5], [0,5,4],       # side +X -X pairings
                [1,2,6], [1,6,5],
                [2,3,7], [2,7,6],
                [3,0,4], [3,4,7],
            ], dtype=int)
            ps.register_surface_mesh("support_box", verts_box, faces_box, color=np.array([0.3,0.3,0.3]), smooth_shade=True)

        loaded_meshes = 0
        for j in range(skel.getNumBodyNodes()):
            bn = skel.getBodyNode(j)
            body_name = bn.getName()
            mesh_entries = official_mesh_map.get(body_name, [])
            for entry in mesh_entries:
                mesh_path = entry['mesh_path']
                if mesh_path and os.path.exists(mesh_path):
                    try:
                        mesh = pv.read(mesh_path)
                        mesh_tri = mesh.triangulate()
                        verts = np.asarray(mesh_tri.points)
                        # Apply local scale
                        verts *= np.array(entry['scale'])
                        # Apply local rotation (XYZ Euler, radians from .osim)
                        rx, ry, rz = entry['rotation']
                        rot_mat = _euler_body_fixed_xyz_to_matrix(rx, ry, rz)
                        verts = verts @ rot_mat.T
                        # Apply local translation
                        verts += np.array(entry['translation'])
                        # Apply body node world transform
                        tf = bn.getWorldTransform()
                        R_bn = np.array(tf.rotation())
                        t_bn = np.array(tf.translation())
                        verts_transformed = np.dot(verts, R_bn.T) + t_bn
                        faces_raw = mesh_tri.faces
                        n_faces = len(faces_raw) // 4
                        tri_faces = faces_raw.reshape(n_faces, 4)[:, 1:4]
                        mesh_file_basename = os.path.basename(mesh_path)
                        mesh_name = f"{body_name}_{mesh_file_basename}_pose_{i}"
                        surf = ps.register_surface_mesh(mesh_name, verts_transformed, tri_faces, color=default_color, smooth_shade=True)
                        loaded_meshes += 1
                        if use_color_gradient and num_poses > 1 and use_alpha_gradient:
                            t_alpha = i / float(max(1, num_poses - 1))
                            alpha = float(start_alpha * (1.0 - t_alpha) + end_alpha * t_alpha)
                        else:
                            alpha = 1.0
                        try:
                            surf.set_transparency(alpha)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Error loading mesh {mesh_path} for body {body_name}: {e}")
        if loaded_meshes == 0:
            print(f"No body meshes loaded for pose {i}. Showing fallback geometry.")
            pts, edges = skeleton_world_points(skel)
            sphere_radius = 0.02
            for pt_idx, pt in enumerate(pts):
                verts = np.array([
                    [0, sphere_radius, 0],
                    [sphere_radius, 0, 0],
                    [0, 0, sphere_radius],
                    [-sphere_radius, 0, 0],
                    [0, 0, -sphere_radius],
                    [0, -sphere_radius, 0]
                ], dtype=float) + pt
                faces = np.array([
                    [0,1,2], [0,2,3], [0,3,4], [0,4,1],
                    [5,2,1], [5,3,2], [5,4,3], [5,1,4]
                ], dtype=int)
                mesh_name = f"joint_{pt_idx}_pose_{i}"
                ps.register_surface_mesh(mesh_name, verts, faces, color=default_color, smooth_shade=True)
            cyl_radius = 0.01
            for edge in edges:
                p0 = pts[edge[0]]
                p1 = pts[edge[1]]
                direction = p1 - p0
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                mid = (p0 + p1) / 2
                perp = np.array([direction[1], -direction[0], 0])
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.array([0, direction[2], -direction[1]])
                perp = perp / np.linalg.norm(perp) * cyl_radius
                verts = np.array([
                    p0 + perp, p0 - perp,
                    p1 + perp, p1 - perp
                ], dtype=float)
                faces = np.array([
                    [0,1,2], [1,3,2]
                ], dtype=int)
                mesh_name = f"bone_{edge[0]}_{edge[1]}_pose_{i}"
                ps.register_surface_mesh(mesh_name, verts, faces, color=default_color, smooth_shade=True)

    # Build trajectory only for selected motion segment
    if show_trajectory:
        model_full = nimble.RajagopalHumanBodyModel()
        skel_full = model_full.skeleton

        # Interpolate offsets only for selected segment
        seg_len = end_idx - start_idx + 1
        full_offsets = np.zeros(seg_len, dtype=float)
        if len(frame_indices) > 0:
            sorted_idxs = list(frame_indices)
            first_idx = int(sorted_idxs[0]) - start_idx
            first_off = float(display_offsets.get(int(sorted_idxs[0]), 0.0))
            full_offsets[:first_idx] = first_off

            for a, b in zip(sorted_idxs[:-1], sorted_idxs[1:]):
                a = int(a) - start_idx; b = int(b) - start_idx
                off_a = float(display_offsets.get(int(a + start_idx), 0.0))
                off_b = float(display_offsets.get(int(b + start_idx), 0.0))
                for k in range(a, b + 1):
                    t = 0.0 if b == a else (k - a) / (b - a)
                    full_offsets[k] = (1.0 - t) * off_a + t * off_b

            last_idx = int(sorted_idxs[-1]) - start_idx
            last_off = float(display_offsets.get(int(sorted_idxs[-1]), 0.0))
            if last_idx < seg_len:
                full_offsets[last_idx:] = last_off

        traj_pts = []
        for f in range(start_idx, end_idx + 1):
            p = expanded[f].copy()
            if p.shape[0] > 3:
                p[3] += full_offsets[f - start_idx]
            skel_full.setPositions(p)
            node = skel_full.getBodyNode(trajectory_body_node)
            if node is not None:
                tf = node.getWorldTransform()
                traj_pts.append(np.array(tf.translation(), dtype=float))

        if len(traj_pts) > 1:
            traj_pts = np.asarray(traj_pts)
            traj_edges = np.array([[i, i + 1] for i in range(len(traj_pts) - 1)], dtype=int)
            traj = ps.register_curve_network(
                f"trajectory_{trajectory_body_node}", traj_pts, traj_edges, color=np.array([1.0, 0.0, 0.0])
            )
            try:
                traj.set_radius(trajectory_line_radius)
            except Exception:
                pass

    # Show Polyscope viewer
    ps.show()


if __name__ == "__main__":
    main()
