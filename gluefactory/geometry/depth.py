import kornia
import torch

from .utils import get_image_coords
from .wrappers import Camera

# # ENTIRELY CHANGED
# def sample_fmap(pts, fmap):
#     h, w = fmap.shape[-2:]
#     grid_sample = torch.nn.functional.grid_sample
#     # pts should be normalized and reshaped to [N, H_out, W_out, 2]
#     pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1).unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 2]
#     # Ensure the fmap is in the correct shape [N, C, H, W]
#     if fmap.ndimension() == 3:
#         fmap = fmap.unsqueeze(1)
#     interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
#     interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
#     return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(0, 2, 1)


def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample

    ### I CHANGED THIS CODE 30-05-24
    # Print shapes for debugging
    print(f"Original pts shape: {pts.shape}")
    print(f"Original fmap shape: {fmap.shape}")

    # Normalize points to the range [-1, 1]
    pts = (pts / torch.tensor([w, h], device=pts.device) * 2 - 1)
    print(f"Normalized pts shape: {pts.shape}")

    # Ensure pts is of shape [N, H_out, W_out, 2]
    pts = pts.view(pts.size(0), 1, -1, 2)  # Reshape to [N, 1, num_keypoints, 2]
    print(f"Reshaped pts shape: {pts.shape}")

    # Ensure fmap is in the correct shape [N, C, H, W]
    # if fmap.ndimension() == 3:
    #     #     fmap = fmap.unsqueeze(1)  # Add channel dimension if missing
    #     # print(f"Adjusted fmap shape: {fmap.shape}")
    # Ensure fmap is in the correct shape [N, C, H, W]
    # Ensure fmap is in the correct shape [N, C, H, W]
    if fmap.ndimension() == 3:
        fmap = fmap.unsqueeze(1)  # Add channel dimension if missing
    print(f"Adjusted fmap shape: {fmap.shape}")

    fmap = fmap.squeeze(2)  # Remove the extra dimension
    print(f"Adjusted fmap shape after squeeze: {fmap.shape}")

    ###

    # pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1)[:, None]
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")

    print(f"interp_lin shape: {interp_lin.shape}")
    print(f"interp_nn shape: {interp_nn.shape}")

    # return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin).squeeze(2)
    ### I CHANGED THIS CODE 30-05-24

    # Handle NaNs by using nearest neighbor interpolation where linear interpolation results in NaNs
    result = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin).squeeze(2)
    print(f"Final result shape: {result.shape}")

    return result

    # return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(0, 2, 1)


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, depth_.new_tensor(float("nan")))
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    validi,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}
    print("\nkpi_3d_i  camera_i type\n", type(camera_i))

    print("\nIn project function:")
    print("kpi.shape:", kpi.shape)
    print("di.shape:", di.shape)
    print("depthj.shape:", depthj.shape if depthj is not None else "None")
    print("camera_i type:", type(camera_i))
    print("camera_j type:", type(camera_j))
    print("T_itoj type:", type(T_itoj))

    kpi_3d_i = camera_i.image2cam(kpi)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    kpi_j, validj = camera_j.cam2image(kpi_3d_j)
    # di_j = kpi_3d_j[..., -1]
    validi = validi & validj

    if depthj is None or ccth is None:
        return kpi_j, validi & validj

    # circle consistency
    dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
    kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
    kpi_j_i, validj_i = camera_i.cam2image(T_itoj.inv().transform(kpi_j_3d_j))
    consistent = ((kpi - kpi_j_i) ** 2).sum(-1) < ccth
    visible = validi & consistent & validj_i & validj
    # visible = validi
    print("kpi_3d_i.shape:", kpi_3d_i.shape)
    print("kpi_3d_j.shape:", kpi_3d_j.shape)
    print("kpi_j.shape:", kpi_j.shape)
    print("validi.shape:", validi.shape)
    print("validj.shape:", validj.shape)
    print("validj_depth.shape:", validj.shape)
    print("kpi_j_3d_j.shape:", kpi_j_3d_j.shape)
    print("kpi_j_i.shape:", kpi_j_i.shape)
    print("consistent:", consistent)
    print("visible.shape:", visible.shape)

    return kpi_j, visible


def dense_warp_consistency(
    depthi: torch.Tensor,
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: Camera,
    cameraj: Camera,
    **kwargs,
):
    kpi = get_image_coords(depthi).flatten(-3, -2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir = project(kpi, di, depthj, camerai, cameraj, T_itoj, validi, **kwargs)

    return kpir.unflatten(-2, depthi.shape[-2:]), validir.unflatten(
        -1, (depthj.shape[-2:])
    )
