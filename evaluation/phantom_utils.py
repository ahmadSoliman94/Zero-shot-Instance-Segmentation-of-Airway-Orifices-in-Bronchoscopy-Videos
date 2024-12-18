import cv2
import numpy as np
import torch
from matplotlib import cm, colors
from numpy.ma import masked_where
from scipy.signal import peak_prominences, find_peaks
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn import cluster
from torch.nn import functional as F


# https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(h: int, w: int, center: (float, float) = None, radius: float = None, exclude_border=False):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center > radius
    # make sure that all borders are excluded
    if exclude_border:
        mask[:, 0] = True
        mask[0, :] = True
        mask[:, -1] = True
        mask[-1, :] = True
    return mask


def calculate_z_threshold_prominence(z_img: np.ndarray, masked: bool = False):
    def find_local_max(s):
        coordinate, peak_property = find_peaks(s, height=(None, None), prominence=(None, None), rel_height=(None, None))
        return coordinate, peak_property

    assert z_img.ndim == 2
    if not masked:
        z_img = np.ma.masked_where(create_circular_mask(*z_img.shape), z_img)
    x, x_peak_property = find_local_max(z_img.mean(axis=0))
    y, y_peak_property = find_local_max(z_img.mean(axis=1))
    return (x, y), (x_peak_property, y_peak_property)


def calculate_z_threshold(z_img: np.ndarray, masked: bool = False):
    def argmax_in_bound(x: np.ndarray, idx: int, radius: int):
        assert x.ndim == 1
        return np.argmax(x[max(0, idx - radius): min(len(x), idx + radius)])

    assert z_img.ndim == 2
    if not masked:
        z_img = np.ma.masked_where(create_circular_mask(*z_img.shape), z_img)

    (y, x) = np.unravel_index(np.argmax(z_img, axis=None), z_img.shape)

    prominence = (peak_prominences(z_img.mean(axis=0),
                                   [argmax_in_bound(z_img.mean(axis=0), x, int(z_img.shape[0] * 1))])[0],
                  peak_prominences(z_img.mean(axis=1),
                                   [argmax_in_bound(z_img.mean(axis=1), y, int(z_img.shape[1] * 1))])[0])
    return (x, y), prominence


class ZBlobThresholdExtractor:
    def __init__(self, spatial_dim: int, percentile: int = 5):
        """
        Uses a blob detector to calculate the threshold via the 8th percentile accept only quadratic images
        :param spatial_dim: int that defines height and width of the image
        """
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.minThreshold = 0.1
        detector_params.thresholdStep = 0.01

        detector_params.filterByArea = True
        detector_params.maxArea = spatial_dim ** 2 / 4

        self.detector = cv2.SimpleBlobDetector_create(detector_params)
        self.circular_mask = create_circular_mask(spatial_dim, spatial_dim)
        self.percentile = percentile

    def determine_z_threshold(self, img: np.ndarray, invert_img: bool = True):
        """

        :param img: image in range [0, 1]
        :param invert_img: invert image, because the blob detector search for black objects
        :return:
        """
        if img.ndim > 2:
            raise NotImplementedError("Only grayscaled images are supported.")
        assert img.shape[0] == img.shape[1]

        img_byte = (img * 255).astype(np.uint8)
        if invert_img:
            img_byte = cv2.bitwise_not(img_byte)
        kpts = self.detector.detect(img_byte)

        # process blob detections
        cover_by_blobs_mask = np.ones_like(img).astype(bool)
        for k in kpts:
            (x, y) = k.pt
            blob_diameter = k.size
            current_blob_mask = create_circular_mask(img.shape[0], img.shape[1],
                                                     center=(x, y),
                                                     radius=blob_diameter / 2)
            cover_by_blobs_mask = np.bitwise_and(cover_by_blobs_mask, current_blob_mask)
        cover_by_blobs_mask = np.bitwise_or(cover_by_blobs_mask, self.circular_mask)

        if not cover_by_blobs_mask.all():
            z_threshold = np.percentile(masked_where(cover_by_blobs_mask, img).compressed(), self.percentile)
            seg_mask = masked_where(img <= z_threshold, np.ones_like(cover_by_blobs_mask))
        else:
            z_threshold = float('nan')
            seg_mask = np.empty_like(img).fill(np.nan)

        return z_threshold, seg_mask, cover_by_blobs_mask


class GroundtruthExtractor:
    def __init__(self, mode: str, spatial_dim: int):
        self.extractor = ZBlobThresholdExtractor(spatial_dim)
        self.mode = mode

    def __call__(self, z: torch.Tensor):
        z = z.squeeze().numpy()
        if self.mode == 'seg':
            # blob + percentile for dilation
            _, seg_mask, _ = self.extractor.determine_z_threshold(z)
            seg_mask = torch.from_numpy(seg_mask.filled(False))
            return seg_mask
        elif self.mode == 'blob':
            _, _, blob_mask = self.extractor.determine_z_threshold(z)
            blob_mask = torch.from_numpy(~blob_mask)
            return blob_mask
        else:
            raise NotImplementedError(f"{self.mode} is not supported.")


def create_color_code(labels: np.ndarray, cmap_id: str = 'Set1'):
    assert labels.ndim == 1

    color_code = cm.get_cmap(cmap_id)(labels)
    color_code[labels == -1] = colors.to_rgba('dimgray', 1)

    return color_code


class ZSegmentationExtractor:
    def __init__(self, spatial_dim: int, avg_pool_kernel_size: int = 3, watershed_compactness: int = 1,
                 out_of_bounds_flag=-1):
        """
        use k-means and marker controlled compact watershed to extract instance airway segmentation from depth img
        :param spatial_dim: spatial size of quadratic depth image
        :param avg_pool_kernel_size: DEPRECATED
        :param watershed_compactness: [0, inf]
        """
        self.spatial_dim = spatial_dim
        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.c = watershed_compactness
        self.bounds_flag = out_of_bounds_flag
        print('out of bounds flag:', out_of_bounds_flag)

        self.circular_mask = torch.from_numpy(~create_circular_mask(spatial_dim, spatial_dim, exclude_border=True))
        self.grid_x, self.grid_y = torch.meshgrid(torch.linspace(0, spatial_dim, spatial_dim),
                                                  torch.linspace(0, spatial_dim, spatial_dim),
                                                  indexing='ij')
        # tensor, that holds the linear index to the depth image's spatial dimensions
        self.airways_label_linear_idx = torch.arange(spatial_dim ** 2).reshape((spatial_dim, spatial_dim))

    def extract_segmentation(self, z_img: torch.Tensor, return_plot_data: bool = False):
        assert z_img.dim() == 2
        assert z_img.shape[-2] == z_img.shape[-1] == self.spatial_dim

        # create feature matrix
        F = torch.stack([self.grid_x, self.grid_y, z_img], dim=-1)
        F = F[self.circular_mask]

        # extract seed for kmeans
        (z_max, z_min) = (torch.max(F[:, 2]), torch.min(F[:, 2]))
        z_seed = torch.stack([z_min, z_max], dim=0).unsqueeze(1)

        # determine which label is fore- and background
        z_labels = cluster.KMeans(  # choose KMeans over MinibatchedKMeans due to its higher stability
            n_clusters=2, init=z_seed, n_init=1
        ).fit_predict(F[:, 2].unsqueeze(1).numpy())
        is_airway = z_labels == 1  # cluster index/label init with z_max → airway

        # consider only airways region
        F_airway = F[is_airway]

        # extract local peaks of smoothed depth image as markers for the watershed
        smoothed_z_img = self.smooth_img(z_img)
        airway_markers = torch.from_numpy(
            peak_local_max(smoothed_z_img.numpy(), min_distance=round(self.spatial_dim * 0.05),
                           threshold_abs=F_airway[:, 2].min().numpy()))
        airway_markers_z = smoothed_z_img[airway_markers.chunk(smoothed_z_img.dim(), dim=1)]  # get z
        airway_markers = torch.cat([airway_markers[:, :2], airway_markers_z], dim=1)

        # sort peaks after xy-coordinates
        coord_idx = np.lexsort((airway_markers[:, 0], airway_markers[:, 1]))
        airway_markers = airway_markers[coord_idx]

        # old approach. Keeping for demonstration of cluster problem
        # airways_labels = cluster.KMeans(
        #     n_clusters=len(seeds), init=seeds, n_init=1,
        # ).fit_predict(F_airway.numpy())

        # convert markers in marker image
        airway_centroids_marker = np.zeros_like(z_img, dtype=int)
        for i, c_idx in enumerate(airway_markers):
            c_idx = tuple(c_idx[:2].int())
            airway_centroids_marker[c_idx] = i + 1  # because foreground already 0
        airways_labels = watershed(-z_img.numpy(), markers=airway_centroids_marker,
                                   mask=(z_img > F_airway[:, 2].min()).numpy(), compactness=self.c)
        airways_labels[~self.circular_mask] = self.bounds_flag
        seg_mask = torch.from_numpy(airways_labels)

        z_mask = torch.full_like(z_img, -1, dtype=torch.int)
        z_mask[self.circular_mask] = torch.from_numpy(z_labels)

        if not return_plot_data:
            return seg_mask.to(torch.int8), z_mask.to(torch.int8)
        else:
            return {
                'smoothed_z_img': smoothed_z_img,
                'airway_centroids': airway_markers,
                'F1': F_airway,
                'F': F,
                'z_labels': z_labels,
                'airways_labels': airways_labels,
                'is_airway': is_airway,
                'seg_mask': seg_mask
            }

    def batched_mahalanobis(self, x: torch.Tensor, y: torch.Tensor, Sigma: torch.Tensor):
        """
        calculate for each point in x the mahalanobis distance to each point in y given the not normalized covariance matrix
        :param x: NxD
        :param y: MxD
        :param Sigma: DxD
        :return: NxM
        """
        assert Sigma.dim() == 2 and Sigma.shape[0] == Sigma.shape[1]
        assert x.dtype == y.dtype == Sigma.dtype == torch.float32  # bmm only implemented for float32
        s_inv = torch.inverse(Sigma)

        distance = x.unsqueeze(-1) - y.T.unsqueeze(0)  # NxMxD
        md = torch.bmm(distance.transpose(-1, -2), s_inv.unsqueeze(0).expand(len(distance), -1, -1))  # NxDxM
        md = torch.bmm(md, distance)  # NxMxM
        md = md.diagonal(dim1=-2, dim2=-1).sqrt()  # NxM

        return md

    def is_in_AABB(self, x: torch.Tensor, c: torch.Tensor):
        """
        checks if points c are in the axis aligned bounding box spanned by x
        :param x: NxD
        :param c: MxD
        :return: boolean tensor with length M
        """

        bounding_box_min = torch.min(x, dim=0, keepdim=True)[0]
        bounding_box_max = torch.max(x, dim=0, keepdim=True)[0]
        is_in_bb = (bounding_box_min.expand(len(c), -1) <= c)
        is_in_bb = is_in_bb.bitwise_and(c <= bounding_box_max.expand(len(c), -1))
        is_in_bb = torch.all(is_in_bb, dim=1)

        return is_in_bb

    def smooth_img(self, img: torch.Tensor):
        """
        smooth given image using efficient box filter as gaussian approximations
        :param img:
        :return:
        """

        def box_filter(x, n):
            assert self.avg_pool_kernel_size % 2 == 1  # for calculating padding
            for _ in range(n):
                x = F.avg_pool2d(x, kernel_size=self.avg_pool_kernel_size, padding=self.avg_pool_kernel_size // 2,
                                 stride=1)
            return x

        assert img.dim() == 2
        img = img.view(1, 1, self.spatial_dim, self.spatial_dim)
        img = box_filter(img, 3)
        return img.squeeze()
