"""
Depth + normal splatter
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM

from gsplat.strategy import DefaultStrategy

try:
    from gsplat.rendering import rasterization_2dgs
except ImportError:
    print("Please install gsplat>=1.0.0")
# from gsplat import rasterize_gaussians
from gsplat.cuda._torch_impl import _quat_to_rotmat as quat_to_rotmat

# from gsplat.cuda_legacy._wrapper import num_sh_bases
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.model_components.lib_bilagrid import BilateralGrid
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.splatfacto import random_quat_tensor, num_sh_bases

@dataclass
class Splatfacto2DGSModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: Splatfacto2DGSModel)

    ### Splatfacto 2dgs configs ###
    
    """Encourage 2D Gaussians"""
    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    ### Splatfacto configs ###
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 5.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    stop_split_at: int = 15000
    """stop splitting at this step"""
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off")
    )
    """Config of the camera optimizer to use"""
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""

    # pearson depth loss lambda
    pearson_lambda: float = 0
    """Regularizer for pearson depth loss"""

    # Model for 2dgs.
    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.01
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002  # 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01  # 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1  # 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000  # 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False


class Splatfacto2DGSModel(SplatfactoModel):
    """Depth + Normal splatter"""

    config: Splatfacto2DGSModelConfig

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )
        
        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=self.config.cull_alpha_thresh,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            prune_scale3d=self.config.cull_scale_thresh,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=self.config.refine_start_iter,
            refine_stop_iter=self.config.refine_stop_iter,
            reset_every=self.config.reset_every,
            refine_every=self.config.refine_every,
            absgrad=False,
            revised_opacity=False,
            key_for_gradient="gradient_2dgs",
        )
        
        self.strategy_state = self.strategy.initialize_state()


    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.

            return {
                name: [self.gauss_params[name]]
                for name in [
                    "means",
                    "scales",
                    "quats",
                    "features_dc",
                    "features_rest",
                    "opacities",
                ]
            }
    def get_outputs(
        self, camera: Cameras
    ) -> Dict[str, Union[torch.Tensor, List[Tensor]]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        self.camera = camera
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds


        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        render_mode = "RGB+ED"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None


        kwargs = {
            "near_plane": 0.2,
            "far_plane": 200,
            "render_mode": render_mode,
            "distloss": self.config.dist_loss,
            "sh_degree": self.config.sh_degree,
        }
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            self.info,
        ) = rasterization_2dgs(
            means=means_crop,  # [N,3]
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),  # [N,4]
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,  # [N,16,3]
            viewmats=viewmat,  # [C, 4, 4]
            Ks=K,  # [C, 3, 3]
            width=W,
            height=H,
            packed=False,
            absgrad=True,
            sparse_grad=False,
            **kwargs,
        )
        if self.training and self.info["means2d"].requires_grad:
            self.info["means2d"].retain_grad()

        self.xys = self.info["means2d"]  # [1, N, 2]
        self.radii = self.info["radii"][0]  # [N]
        self.num_tiles_hit = self.info["tiles_per_gauss"]

        colors = render_colors[..., 0:3]
        alpha = render_alphas
        background = self._get_background_color()
        rgb = colors + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        depth_im = render_colors[..., 3:4]
        depth_im = torch.where(
            alpha > 0, depth_im, depth_im.detach().max()
        ).squeeze(0)

        normals_im = render_normals[0]  # [-1,1]
        # convert normals from [-1,1] to [0,1]
        normals_im = normals_im / normals_im.norm(dim=-1, keepdim=True)
        normals_im = (normals_im + 1) / 2

        surface_normal = normals_from_depth  # [-1,1]
        # convert normals from [-1,1] to [0,1]
        surface_normal = (surface_normal + 1) / 2

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "normal": normals_im,  # predicted normal from gaussians
            "surface_normal": surface_normal,  # normal from surface / depth
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = super().get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict
        )
        main_loss = loss_dict["main_loss"]
        scale_reg = loss_dict["scale_reg"]


        return {"main_loss": main_loss, "scale_reg": scale_reg}

    


