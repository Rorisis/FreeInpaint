import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage
import scipy


def normalize(x):
    return x / (np.linalg.norm(x) + 1e-8)


def viewmatrix(lookdir, up, position):
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def pad_poses(p):
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    return p[..., :3, :4]


def focus_point_fn(poses):
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def transform_poses_pca(poses):
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform


def get_traj_ellipse(poses_4x4, n_frames=120, z_variation=0.0):
    """
    Generate an orbit/elliptical trajectory from reference camera poses.
    poses_4x4: [N, 4, 4] NumPy array of OpenCV camera-to-world matrices.
    """
    # ==========================================
    # Convert coordinates from OpenCV to OpenGL.
    # Multiply by a diagonal matrix to flip the Y and Z axes.
    # ==========================================
    flip_mat = np.diag([1, -1, -1, 1])
    poses_gl = poses_4x4 @ flip_mat
    poses = poses_gl[:, :3, :4]

    # 1. Apply PCA and center the poses in OpenGL coordinates.
    poses_recentered, colmap_to_world_transform = transform_poses_pca(poses)

    # 2. Find the cameras' focus point (approximate object center).
    center = focus_point_fn(poses_recentered)
    offset = np.array([center[0], center[1], 0])

    # Compute the extents of the ellipse axes.
    sc = np.percentile(np.abs(poses_recentered[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset

    # Compute the height range along the Z axis.
    z_low = np.percentile((poses_recentered[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses_recentered[:, :3, 3]), 90, axis=0)

    # 3. Generate points along the ellipse.
    def get_positions(theta):
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation
                * (z_low[2] + (z_high - z_low)[2] * (np.cos(theta) * 0.5 + 0.5)),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)[:-1]

    # Compute the average up vector.
    avg_up = poses_recentered[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    # 4. Construct look-at matrices.
    # Here p - center points the Z axis away from the center, as in OpenGL.
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])

    # Expand to homogeneous 4x4 matrices.
    new_poses_4x4 = pad_poses(new_poses)

    # Transform back to the original world frame, still in OpenGL coordinates.
    new_poses_orig_gl = np.linalg.inv(colmap_to_world_transform) @ new_poses_4x4

    # ==========================================
    # Convert coordinates from OpenGL to OpenCV.
    # Flip the Y and Z axes again to match DA3's input convention.
    # ==========================================
    new_poses_orig_cv = new_poses_orig_gl @ flip_mat

    return new_poses_orig_cv


def generate_spiral_trajectory(
    c2ws_opencv, pts_3d, n_frames=120, radius_mult=1.0, spirals=1, add_3d_bobbing=True
):
    """
    Generate an adaptive spiral camera path (OpenCV: X right, Y down, Z forward).

    Args:
    c2ws_opencv: (N, 4, 4) input camera-to-world matrices.
    pts_3d: (M, 3) point cloud used to estimate the scene focus.
    n_frames: Number of trajectory frames.
    radius_mult: Trajectory radius multiplier.
    spirals: Number of spiral revolutions.
    add_3d_bobbing: Add small forward/backward motion along Z for a 3D spiral.
    """
    centers = c2ws_opencv[:, :3, 3]
    Y_axes = c2ws_opencv[:, :3, 1]  # Down
    Z_axes = c2ws_opencv[:, :3, 2]  # Forward

    # =========================================================
    # 1. Construct an average camera robust to panoramic and extreme viewpoints.
    # =========================================================
    mean_center = np.mean(centers, axis=0)

    # Average the Z axes, guarding against cancellation for 360-degree captures.
    avg_Z = np.mean(Z_axes, axis=0)
    norm_Z = np.linalg.norm(avg_Z)
    if norm_Z < 1e-5:
        # If directions cancel, face the origin or fall back to [0, 0, 1].
        avg_Z = (
            -mean_center
            if np.linalg.norm(mean_center) > 1e-5
            else np.array([0.0, 0.0, 1.0])
        )
        norm_Z = np.linalg.norm(avg_Z)
    avg_Z = avg_Z / norm_Z

    # Average the Y axes, guarding against opposing directions.
    avg_Y_temp = np.mean(Y_axes, axis=0)
    if np.linalg.norm(avg_Y_temp) < 1e-5:
        avg_Y_temp = np.array([0.0, 1.0, 0.0])

    # Apply Gram-Schmidt orthogonalization.
    avg_X = np.cross(avg_Y_temp, avg_Z)
    norm_X = np.linalg.norm(avg_X)
    if norm_X < 1e-5:  # Guard against parallel Y_temp and Z vectors.
        avg_Y_temp = np.array([1.0, 0.0, 0.0])
        avg_X = np.cross(avg_Y_temp, avg_Z)
        norm_X = np.linalg.norm(avg_X)
    avg_X = avg_X / norm_X

    avg_Y = np.cross(avg_Z, avg_X)
    avg_Y = avg_Y / np.linalg.norm(avg_Y)

    # =========================================================
    # 2. Estimate the scene focus point.
    # =========================================================
    vecs = pts_3d - mean_center
    depths = np.dot(vecs, avg_Z)
    valid_depths = depths[depths > 0.1]  # Exclude points behind the camera.

    if len(valid_depths) > 0:
        focal_depth = np.median(valid_depths)
    else:
        focal_depth = 5.0  # Fallback when the point cloud is empty or all points are behind the camera.

    focus_pt = mean_center + avg_Z * focal_depth

    # =========================================================
    # 3. Estimate and smooth the camera-distribution radii.
    # =========================================================
    rel_centers = centers - mean_center
    dx = np.dot(rel_centers, avg_X)
    dy = np.dot(rel_centers, avg_Y)

    rad_x = np.percentile(np.abs(dx), 90)
    rad_y = np.percentile(np.abs(dy), 90)

    # Enforce a minimum radius.
    if rad_x < 1e-3:
        rad_x = focal_depth * 0.1
    if rad_y < 1e-3:
        rad_y = focal_depth * 0.1 * 0.6

    # Limit the ellipse aspect ratio to avoid an overly flat path and abrupt motion.
    aspect_ratio = rad_x / rad_y
    if aspect_ratio > 2.0:
        rad_y = rad_x / 2.0
    elif aspect_ratio < 0.5:
        rad_x = rad_y / 2.0

    rad_x *= radius_mult
    rad_y *= radius_mult

    # =========================================================
    # 4. Generate the adaptive spiral path.
    # =========================================================
    spiral_c2ws = []
    for i in range(n_frames):
        t = i / n_frames
        # Use linear time sampling for a seamlessly looping spiral.
        theta = 2.0 * np.pi * spirals * t

        # Trace an ellipse in the adaptive plane.
        off_x = rad_x * np.cos(theta)
        off_y = rad_y * np.sin(theta)

        # Add slight forward/backward motion along Z to form a 3D spiral.
        # sin(theta * 2.0) adds two gentle depth oscillations per revolution.
        off_z = focal_depth * 0.05 * np.sin(theta * 2.0) if add_3d_bobbing else 0.0

        # Compute the new camera position.
        pos = mean_center + off_x * avg_X + off_y * avg_Y + off_z * avg_Z

        # Construct a look-at frame aimed at the focus point.
        Z = focus_pt - pos
        Z = Z / np.linalg.norm(Z)

        X = np.cross(avg_Y, Z)
        X = X / np.linalg.norm(X)

        Y = np.cross(Z, X)
        Y = Y / np.linalg.norm(Y)

        # Assemble the 4x4 camera-to-world matrix.
        c2w = np.eye(4)
        c2w[:3, 0] = X
        c2w[:3, 1] = Y
        c2w[:3, 2] = Z
        c2w[:3, 3] = pos

        spiral_c2ws.append(c2w)

    return np.stack(spiral_c2ws)


def generate_orbit_trajectory(
    c2ws_opencv, pts_3d, n_frames=120, elevation_deg=5.0, radius_mult=1.5
):
    """
    Generate a smooth orbit centered on the 3D point cloud.
    radius_mult: Distance multiplier; 1.0 preserves distance and 1.5 increases it by 50%.
    """
    # 1. Use the first camera pose as the seed.
    seed_c2w = c2ws_opencv[0].copy()
    seed_pos = seed_c2w[:3, 3]
    seed_R = seed_c2w[:3, :3]

    # 2. Estimate the scene focus point.
    focus_pt = np.median(pts_3d, axis=0)

    # 3. Determine the up vector / rotation axis.
    all_y = c2ws_opencv[:, :3, 1]
    up_axis = -np.mean(all_y, axis=0)
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-6)

    # 4. Generate the orbit.
    orbit_c2ws = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames

        # A. Construct a 3x3 rotation matrix about the up axis.
        rot_vec = up_axis * angle
        R_orbit = Rotation.from_rotvec(rot_vec).as_matrix()

        # B. Rotate the camera position around focus_pt and scale its distance.
        # Multiplying by radius_mult moves the camera along the viewing direction.
        rel_pos = (seed_pos - focus_pt) * radius_mult
        new_pos = focus_pt + R_orbit @ rel_pos

        # C. Rotate the camera orientation to preserve its gaze.
        new_R = R_orbit @ seed_R

        # D. Optionally apply elevation.
        if elevation_deg != 0.0:
            tilt = np.radians(elevation_deg)
            # Rotate about the camera's local X axis.
            R_tilt = Rotation.from_rotvec(new_R[:, 0] * tilt).as_matrix()
            new_R = R_tilt @ new_R
            # Adjust camera height using the distance already scaled by radius_mult.
            current_radius = np.linalg.norm(rel_pos)
            new_pos += up_axis * (current_radius * np.sin(tilt))

        # E. Assemble the camera pose.
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = new_R
        new_c2w[:3, 3] = new_pos
        orbit_c2ws.append(new_c2w)

    return np.stack(orbit_c2ws)


def generate_robust_trajectory(
    c2ws_opencv, n_frames=120, radius_mult=1.0, elevation_deg=0.0
):
    """
    Generate a smooth orbit from predicted poses in OpenCV C2W format.
    Supports both 360-degree and forward-facing scenes.

    c2ws_opencv: [V, 4, 4] NumPy array of predicted camera-to-world matrices.
    """
    centers = c2ws_opencv[:, :3, 3]
    Z_axes = c2ws_opencv[:, :3, 2]  # OpenCV Z points forward.
    Y_axes = c2ws_opencv[:, :3, 1]  # OpenCV Y points down.

    # 1. Determine the world up direction.
    # Negate the mean camera Y (down) axis to obtain up.
    avg_down = np.mean(Y_axes, axis=0)
    up_vector = -avg_down
    up_vector = up_vector / np.linalg.norm(up_vector)

    # 2. Determine the focus point.
    mean_center = np.mean(centers, axis=0)

    # Distinguish forward-facing from 360-degree captures.
    # Low variance in camera Z directions indicates forward-facing views.
    z_std = np.mean(np.std(Z_axes, axis=0))

    mean_dist = np.mean(np.linalg.norm(centers - mean_center, axis=1))
    if mean_dist < 1e-3:
        mean_dist = 1.0  # fallback

    if z_std < 0.2:
        # Forward-facing scene: place the focus in front of the cameras.
        focus_pt = mean_center + np.mean(Z_axes, axis=0) * mean_dist * 2.0
    else:
        # Object-centric 360-degree scene: use the camera centroid as the focus.
        focus_pt = mean_center

    # 3. Determine the orbit radius.
    radius = mean_dist * radius_mult

    # 4. Construct the orbit basis (U, V).
    # U: project the first camera's position offset onto the horizontal plane.
    v_start = centers[0] - focus_pt
    v_start_h = v_start - np.dot(v_start, up_vector) * up_vector

    # Guard against a zero U vector when the first camera is directly overhead.
    if np.linalg.norm(v_start_h) < 1e-4:
        v_start_h = np.array([1.0, 0.0, 0.0])
        v_start_h = v_start_h - np.dot(v_start_h, up_vector) * up_vector

    u_axis = v_start_h / np.linalg.norm(v_start_h)
    # V is perpendicular to both up and U.
    v_axis = np.cross(up_vector, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    # 5. Generate the trajectory.
    h_offset = radius * np.sin(np.radians(elevation_deg))
    r_eff = radius * np.cos(np.radians(elevation_deg))

    render_c2ws = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames

        # Camera position.
        pos = (
            focus_pt
            + r_eff * (np.cos(angle) * u_axis + np.sin(angle) * v_axis)
            + h_offset * up_vector
        )

        # Camera orientation in OpenCV coordinates: X right, Y down, Z forward.
        Z = focus_pt - pos
        Z = Z / np.linalg.norm(Z)

        X = np.cross(up_vector, Z)
        X = X / np.linalg.norm(X)

        Y = np.cross(X, Z)  # Cross the right and forward axes to obtain down.
        Y = Y / np.linalg.norm(Y)

        c2w = np.eye(4)
        c2w[:3, 0] = X
        c2w[:3, 1] = Y
        c2w[:3, 2] = Z
        c2w[:3, 3] = pos

        render_c2ws.append(c2w)

    return np.stack(render_c2ws)


def smooth_camera_trajectory(c2ws, sigma=2.0):
    """
    Apply Gaussian smoothing to a camera trajectory to reduce jitter.
    c2ws: [N, 4, 4] NumPy array of camera-to-world matrices.
    sigma: Smoothing strength; larger values can deviate more from the input path. Typical range: 2.0-4.0.
    """
    N = len(c2ws)
    if N < 5:
        return c2ws

    c2ws_smooth = np.copy(c2ws)

    # 1. Smooth translations.
    trans = c2ws[:, :3, 3]  # [N, 3]
    trans_smooth = scipy.ndimage.gaussian_filter1d(trans, sigma=sigma, axis=0)
    c2ws_smooth[:, :3, 3] = trans_smooth

    # 2. Smooth rotations via quaternions.
    rots = Rotation.from_matrix(c2ws[:, :3, :3])
    quats = rots.as_quat()  # [N, 4] (x, y, z, w)

    # Resolve quaternion sign flips to maintain continuity.
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    # Apply Gaussian smoothing to the quaternions.
    quats_smooth = scipy.ndimage.gaussian_filter1d(quats, sigma=sigma, axis=0)
    # Normalize the quaternions.
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)

    # Convert back to rotation matrices.
    rots_smooth = Rotation.from_quat(quats_smooth).as_matrix()
    c2ws_smooth[:, :3, :3] = rots_smooth

    return c2ws_smooth
