import torch
import torch.nn as nn


AP3D_ROOT = 6

AP3D_ROOT_BONE_EDGES = [
    (6, 0), (0, 1), (1, 2),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 10), (10, 11), (11, 12),
    (7, 13), (13, 14), (14, 15)
]


class RootBoneKinematics(nn.Module):
    def __init__(self, num_joints=16, root=AP3D_ROOT, edges=AP3D_ROOT_BONE_EDGES):
        super().__init__()
        self.num_joints = num_joints
        self.root = root
        self.edges = edges
        self.data_dim = num_joints * 3
        self.rep_dim = 3 + len(edges) * 3
        assert self.rep_dim == self.data_dim, f"Root-bone dim {self.rep_dim} must equal Cartesian dim {self.data_dim}"

    def to_xyz(self, x):
        if x.dim() == 4:
            return x
        B, T, D = x.shape
        assert D == self.data_dim, f"Expected D={self.data_dim}, got {D}"
        return x.reshape(B, T, self.num_joints, 3)

    def encode(self, traj):
        xyz = self.to_xyz(traj)
        root_pos = xyz[:, :, self.root]
        bone_vecs = []
        for p, c in self.edges:
            bone_vecs.append(xyz[:, :, c] - xyz[:, :, p])
        bone_vecs = torch.stack(bone_vecs, dim=2)
        rep = torch.cat([root_pos, bone_vecs.reshape(xyz.shape[0], xyz.shape[1], -1)], dim=-1)
        return rep

    def decode(self, rep):
        import numpy as np
        import torch

        is_torch = torch.is_tensor(rep)
        is_numpy = isinstance(rep, np.ndarray)

        if not is_torch and not is_numpy:
            raise TypeError(f"decode expects torch.Tensor or np.ndarray, got {type(rep)}")

        if rep.ndim < 3:
            raise ValueError(f"decode expects at least 3D input, got shape {rep.shape}")

        *prefix_shape, D = rep.shape

        if D != self.rep_dim:
            raise ValueError(f"Expected root-bone dim={self.rep_dim}, got {D}, input shape={rep.shape}")

        root_pos = rep[..., :3]
        bone_vecs = rep[..., 3:].reshape(*prefix_shape, len(self.edges), 3)

        if is_torch:
            xyz = rep.new_zeros(*prefix_shape, self.num_joints, 3)
        else:
            xyz = np.zeros((*prefix_shape, self.num_joints, 3), dtype=rep.dtype)

        xyz[..., self.root, :] = root_pos

        filled = [False for _ in range(self.num_joints)]
        filled[self.root] = True

        remaining = list(enumerate(self.edges))

        while len(remaining) > 0:
            new_remaining = []
            progress = False

            for e, (p, c) in remaining:
                if filled[p]:
                    xyz[..., c, :] = xyz[..., p, :] + bone_vecs[..., e, :]
                    filled[c] = True
                    progress = True
                else:
                    new_remaining.append((e, (p, c)))

            if not progress:
                raise ValueError(
                    f"Cannot decode root-bone representation. Edges are not connected from root={self.root}. Remaining edges={new_remaining}")

            remaining = new_remaining

        return xyz.reshape(*prefix_shape, self.data_dim)

    def bone_vectors(self, traj):
        xyz = self.to_xyz(traj)
        bone_vecs = []
        for p, c in self.edges:
            bone_vecs.append(xyz[:, :, c] - xyz[:, :, p])
        return torch.stack(bone_vecs, dim=2)

    def bone_directions(self, traj, eps=1e-6):
        bones = self.bone_vectors(traj)
        return bones / torch.norm(bones, dim=-1, keepdim=True).clamp_min(eps)