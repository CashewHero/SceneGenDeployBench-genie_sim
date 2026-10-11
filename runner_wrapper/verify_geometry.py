"""CPU-only check of upstream cube extrinsics and the output coordinate metadata."""

import numpy as np
import torch

from geniesim_world.utils.cubes import Equirec2Cube, gen_cubes
from sharp.utils.gaussians import get_unprojection_matrix


def main() -> None:
    size = 65
    projection = Equirec2Cube(64, 128, size)
    _, _, _, intrinsics, extrinsics = gen_cubes(
        np.zeros((64, 128, 3), dtype=np.uint8),
        np.ones((64, 128), dtype=np.float32),
        np.ones((64, 128), dtype=bool),
        np.eye(4, dtype=np.float32),
        out_size=size,
    )
    for face in range(6):
        intr = torch.eye(4)
        intr[:3, :3] = intrinsics[face]
        unprojection = get_unprojection_matrix(torch.linalg.inv(extrinsics[face]), intr, (size, size))
        for row, col in ((32, 32), (16, 16), (48, 48), (16, 48), (48, 16)):
            gx, gy = projection.grid[row, col]
            ndc = torch.tensor([2 * gx, -2 * gy, 1.0, 1.0], dtype=torch.float32)
            world = (unprojection @ ndc)[:3].numpy()
            # ERP projection.xyz uses right/up/forward. Its negation is the
            # original merged frame: left/down/back, with the input center at -Z.
            expected = -2 * projection.xyz[row, face * size + col]
            np.testing.assert_allclose(world, expected, atol=1e-6)
    print("Verified all six upstream cube faces and LDB output coordinates")


if __name__ == "__main__":
    main()
