import pickle
import os
import math
import scipy
import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider
from matplotlib import cm, colors
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.ticker import PercentFormatter

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from astropy.stats import SigmaClip
from astropy.nddata.utils import Cutout2D

from astrodendro import Dendrogram
from astrodendro.analysis import PPStatistic
from photutils.background import StdBackgroundRMS
from photutils.aperture import CircularAperture
from scipy.ndimage import rotate, shift, gaussian_filter
from scipy.optimize import curve_fit
from pandas import DataFrame as df

import warnings

warnings.simplefilter("ignore")


class ImageUtilities:
    """
    Utilities to get information from HDU, transform HDU, etc.
    """

    @staticmethod
    def make_2d_hdu(data, header):
        # Remove axes w/ a length of 1
        data2d = np.squeeze(data)
        if data2d.ndim != 2:
            raise ValueError(
                f"Expected 2D data after squeeze, got shape {data2d.shape}"
            )

        wcs2d = WCS(header).celestial
        new_header = header.copy()

        # Remove all old WCS cards, then merge in the clean 2D WCS cards
        for key in list(new_header.keys()):
            if key.startswith(
                (
                    "NAXIS",
                    "CTYPE",
                    "CRVAL",
                    "CRPIX",
                    "CDELT",
                    "CUNIT",
                    "CROTA",
                    "PC",
                    "CD",
                    "WCSAXES",
                    "PV",
                )
            ):
                del new_header[key]

        new_header.update(wcs2d.to_header())
        new_header["NAXIS"] = 2
        new_header["NAXIS1"] = data2d.shape[1]
        new_header["NAXIS2"] = data2d.shape[0]

        return fits.PrimaryHDU(data=data2d, header=new_header)

    @staticmethod
    def get_pixel_scale(hdu):
        header = hdu.header
        wcs = WCS(header)
        # Get pixel scale from WCS
        # pixel_scale = np.abs(wcs.pixel_scale_matrix[0, 0]) * u.deg
        if "PIXAR_A2" in header:
            pixel_scale_arcsec = np.sqrt(header["PIXAR_A2"])
        elif "CD1_1" in header:
            pixel_scale_arcsec = np.absolute(header["CD1_1"]) * 3600.0
        elif "CDELT1" in header:
            pixel_scale_arcsec = np.absolute(header["CDELT1"]) * 3600.0

        return pixel_scale_arcsec

    @staticmethod
    def source_in_image(im_shape, pix_coord):  # From Yao-Lun Yang
        # assume the image shape is in (y, x)
        low_x = pix_coord[0] >= 0
        low_y = pix_coord[1] >= 0
        high_x = pix_coord[0] <= im_shape[1]
        high_y = pix_coord[1] <= im_shape[0]
        return low_x & low_y & high_x & high_y

    @staticmethod
    def rotate_hdu(hdu, rotation_angle_deg):

        if hdu.data.ndim > 2:
            hdu_2D = ImageUtilities.make_2d_hdu(hdu.data, hdu.header)
        else:
            hdu_2D = hdu

        data = hdu_2D.data
        header = hdu_2D.header

        # First, rotate the data
        new_data = rotate(data, rotation_angle_deg, reshape=False, order=1, cval=np.nan)

        # Now, copy and adjust the header
        new_header = header.copy()

        # Get image dimensions
        ny, nx = data.shape

        # Get original reference pixel coordinates
        crpix1 = new_header.get("CRPIX1", nx / 2 + 1)  # FITS uses 1-based indexing
        crpix2 = new_header.get("CRPIX2", ny / 2 + 1)

        # Convert to 0-based coordinates for rotation calculation
        crpix1_0based = crpix1 - 1
        crpix2_0based = crpix2 - 1

        # Image center in 0-based coordinates
        center_x = (nx - 1) / 2
        center_y = (ny - 1) / 2

        # Calculate reference pixel position relative to center
        dx = crpix1_0based - center_x
        dy = crpix2_0based - center_y

        # Apply rotation to the reference pixel offset
        angle_rad = np.radians(rotation_angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotate the offset (note: this is the inverse of the image rotation)
        # because we're rotating the coordinate system, not the point
        dx_new = cos_a * dx + sin_a * dy
        dy_new = -sin_a * dx + cos_a * dy

        # Calculate new reference pixel coordinates
        new_crpix1 = center_x + dx_new + 1  # Convert back to 1-based
        new_crpix2 = center_y + dy_new + 1

        # Update CRPIX values
        new_header["CRPIX1"] = new_crpix1
        new_header["CRPIX2"] = new_crpix2

        # Create rotation matrix for PC matrix update
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Get original PC matrix (defaults to identity if not present)
        pc_matrix = np.array(
            [
                [new_header.get("PC1_1", 1.0), new_header.get("PC1_2", 0.0)],
                [new_header.get("PC2_1", 0.0), new_header.get("PC2_2", 1.0)],
            ]
        )

        # Apply rotation: PC_new = PC_old × R
        new_pc_matrix = pc_matrix @ rotation_matrix

        # Update PC matrix in header (CDELT values remain unchanged)
        new_header["PC1_1"] = new_pc_matrix[0, 0]
        new_header["PC1_2"] = new_pc_matrix[0, 1]
        new_header["PC2_1"] = new_pc_matrix[1, 0]
        new_header["PC2_2"] = new_pc_matrix[1, 1]

        # Put the rotated image in a FITS HDU
        new_hdu = fits.ImageHDU(new_data)
        new_hdu.header = new_header

        return new_hdu

    @staticmethod
    def PA_to_rotation_angle(hdu, PA_deg):
        """Calculates the angle by which an image needs to be rotated to align with the provided position angle

        Args:
            hdu (FITS Image HDU): image HDU
            PA_deg (float): position angle

        Returns:
            float: rotation angle
        """
        image_pa = hdu.header["PA_APER"]
        return PA_deg - image_pa - 90

    @staticmethod
    def cutout(hdu, central_coords, x_size_arcsec, y_size_arcsec):
        """
        based on: https://docs.astropy.org/en/stable/nddata/utils.html

        Cuts an image to a specified size and changes the header accordingly.
        Parameters
        ----------
        central_coords: `astropy.coordinates.SkyCoord`
            Central coordinates of the PRIMARY source in the region. This must be a SkyCoord statement.
        size_arcsec: float
            Size (in arcseconds) of the cut out image.

        Returns
        ---------
        hdu: FITS HDU
            HDU containing the cutout image and adapted header.
        """
        # Load the image and the WCS

        if hdu.data.ndim > 2:
            hdu_2D = ImageUtilities.make_2d_hdu(hdu.data, hdu.header)
        else:
            hdu_2D = hdu

        data = hdu_2D.data
        header = hdu_2D.header

        wcs = WCS(header)

        if "CD1_1" in header:
            pixel_scale = np.absolute(header["CD1_1"]) * 3600.0
        elif "CDELT1" in header:
            pixel_scale = np.absolute(header["CDELT1"]) * 3600.0
        else:
            raise Exception("Neither CD1_1 nor CDELT1 were found in the header")

        x_size_pixels = x_size_arcsec / pixel_scale
        y_size_pixels = y_size_arcsec / pixel_scale

        # Make the cutout, including the WCS
        cutout = Cutout2D(
            data,
            position=central_coords,
            size=(y_size_pixels, x_size_pixels),
            wcs=wcs,
        )

        # Put the cutout image in the FITS HDU
        output_hdu = fits.ImageHDU(cutout.data)
        output_hdu.header = header.copy()

        # Update the FITS header with the cutout WCS
        output_hdu.header.update(cutout.wcs.to_header())

        return output_hdu


class PlottingUtilities:
    # Utilities for plotting images
    @staticmethod
    def plot_img(
        image_hdu,
        save_as=None,
        dpi=300,
        ax=None,
        cmap="gray",
        distance_pc=False,
        show=True,
        colorbar=False,
        show_axes_labels=True,
        stretch="sqrt",
        percent=99,
        patches=[],
        source_skycoords=[],
        source_color="black",
        plot_nums=False,
        figsize=None,
        marker_size=5,
        marker="*",
    ):

        # Setup the figure
        # Replace NaNs with zeros
        data = np.nan_to_num(image_hdu.data, copy=True, nan=0.0)
        wcs = WCS(image_hdu.header)

        if figsize is None:
            fig_x = 5 * np.shape(data)[1] / np.shape(data)[0]
            fig_y = 5 * np.shape(data)[0] / np.shape(data)[0]
            figsize = (fig_y, fig_x)

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection=wcs), figsize=figsize)
        else:
            ax = ax
            fig = ax.figure

        # Plot the data
        norm = simple_norm(data, stretch=stretch, percent=percent)
        ax.imshow(data, origin="lower", cmap=cmap, norm=norm)

        if distance_pc is not False:
            scalebar_arcsec = 50
            scalebar_pc = ((scalebar_arcsec * distance_pc * u.au).to(u.pc)).value
            scalebar_pix = scalebar_arcsec / ImageUtilities.get_pixel_scale(image_hdu)
            scalebar_x = np.array(np.linspace(0, scalebar_pix)) + 300  # 300#+ 3500
            scalebar_y = np.array([400] * len(scalebar_x))
            plt.plot(scalebar_x, scalebar_y, color="white")
            plt.text(
                scalebar_x[0] + 10,
                scalebar_y[0] + 100,
                str(np.round(scalebar_pc, 3)) + " pc",
                color="white",
                fontsize=figsize[0] + 2,
            )
            plt.text(
                scalebar_x[0] + int(len(scalebar_x) / 2),
                scalebar_y[0] - 150,
                str(np.round(scalebar_arcsec, 3)) + "''",
                color="white",
                fontsize=figsize[0] + 2,
            )

        # ax.tick_params(axis="both", labelsize=figsize[0] + 2)
        if colorbar:
            fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                orientation="vertical",
                fraction=0.05,
                label="MJy/sr",
            )

        # If patches (like rectangles, etc.) are provided, plot them
        if len(patches) > 0:
            [ax.add_patch(patch) for patch in patches]

        # If sources are provided, plot them
        if len(source_skycoords) > 0:
            source_pixcoords = [wcs.world_to_pixel(coord) for coord in source_skycoords]
            for num, pixcoord in enumerate(source_pixcoords):
                if ImageUtilities.source_in_image(
                    im_shape=np.shape(data), pix_coord=pixcoord
                ):
                    plt.scatter(
                        pixcoord[0],
                        pixcoord[1],
                        marker=marker,
                        s=marker_size,
                        color=source_color,
                    )
                    if plot_nums == True:
                        plt.text(
                            source_pixcoords[num][0] + 10,
                            source_pixcoords[num][1],
                            str(num),
                            fontsize=14,
                            color=source_color,
                        )

        if show_axes_labels == False:
            ax = plt.gca()
            ax.set_axis_off()
        # ax.set_xlabel("")
        # ax.set_xticks([])

        # ax.set_ylabel("")

        else:
            ax.set_xlabel("RA (J2000)")
            ax.set_ylabel("DEC (J2000)")

        if save_as:
            plt.savefig(save_as, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show(fig)

    @staticmethod
    def wallplot(
        image_hdus,
        window_x_arcsec=50.0,
        window_y_arcsec=12.5,
        show=True,
        save_as=None,
        plot_names=None,
        cmap="gray",
        figsize_factor=1,
    ):
        ncols = 2
        nrows = math.ceil(len(image_hdus) / ncols)
        figsize = (
            figsize_factor * window_x_arcsec / window_y_arcsec * ncols,
            figsize_factor * nrows,
        )  # (ncols*window_x_arcsec/window_y_arcsec, nrows)#figsize_factor * nrows, figsize_factor * window_x_arcsec / window_y_arcsec * ncols)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        plt.tight_layout()
        fig.canvas.toolbar_position = (
            "top"  # this also switches the orientation to horizontal, too
        )
        fig.canvas.header_visible = False  # Gets rid of "Figure 1" on top

        # Ticks and labels
        sample_wcs = WCS(image_hdus[0].header)
        sample_pscale = ImageUtilities.get_pixel_scale(image_hdus[0])
        x_ticks = window_x_arcsec * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x_positions = x_ticks / sample_pscale
        x_labels = [str(int(x - (window_x_arcsec / 2))) for x in x_ticks]

        y_ticks = window_y_arcsec * np.array([0.1, 0.5, 0.9])
        y_positions = y_ticks / sample_pscale
        y_labels = [str(int(y - (window_y_arcsec / 2))) for y in y_ticks]

        # Plot each image
        for i, axis in enumerate(ax.reshape(-1)):
            if i < len(image_hdus):
                image_hdu = image_hdus[i]
                norm = simple_norm(image_hdu.data, stretch="sqrt", percent=99)
                axis.imshow(image_hdu.data, origin="lower", cmap=cmap, norm=norm)
            axis.set_xticks(x_positions)
            axis.set_xticklabels([])
            axis.set_yticks(y_positions)
            axis.set_yticklabels([])
            axis.tick_params(axis="both", labelsize=22)

        for row in range(nrows):
            ax[row, 0].set_yticklabels(y_labels)
        for col in range(ncols):
            ax[nrows - 1, col].set_xticklabels(x_labels)

        ax[nrows - 1, 0].set_xticklabels(x_labels)
        ax[nrows - 1, 0].set_yticklabels(y_labels)
        ax[nrows - 1, 0].set_xlabel("Offset [arcsec]", fontsize=22)
        ax[nrows - 1, 0].set_ylabel("Offset [arcsec]", fontsize=22)

        if save_as:
            plt.savefig(save_as, dpi=400, bbox_inches="tight", transparent=True)

        if show:
            plt.show(fig)
        else:
            plt.close(fig)


class Outflow:
    """
    A class used to represent an outflow in a given image

    Args:
        image_hdu (FITS ImageHDU): HDU that contains the image data
        central_coord (astropy.coordinates.sky_coordinate.SkyCoord): SkyCoord representing the central coordinates of the outflow's driving source (or central location)
        rotation_angle (float): Angle (in degrees) by which the image needs to be rotated to make the outflow horizontal
        window_x_arcsec (float, optional): Window size (in the x-dimension) in arcseconds. Defaults to 50.0.
        window_y_arcsec (float, optional): Window size (in the y-dimension) in arcseconds. Defaults to 12.5.
    """

    def __init__(
        self,
        image_hdu,
        central_coord,
        rotation_angle,
        window_x_arcsec=50.0,
        window_y_arcsec=12.5,
    ):
        self.central_coord = central_coord
        self.window_x_arcsec = window_x_arcsec
        self.window_y_arcsec = window_y_arcsec
        self._img_rotation_angle = rotation_angle

        # Get an HDU corresponding to a window of data centered on central_coord and with size (window_x_arcsec, winow_y_arcsec)
        self.window_hdu = self._outflow_region(image_hdu, rotation_angle)

        # These will be determined later
        self._selected_structures = []
        self._selected_structure_data = []
        self._selected_structure_mask = np.zeros(np.shape(self.window_hdu.data))
        self._position_angle = None
        self._position_angle_err = None
        self._dendrogram = None

    def plot_outflow(
        self,
        fig_height=2.0,
        ax=None,
        feature_coords=None,
        patches=[],
        plot_title=None,
        cmap="magma",
        show=True,
        save_as=None,
        dpi=300,
        stretch="sqrt",
        percent=99.0,
        show_xlabels=True,
        show_ylabels=True,
    ):
        """Plots the window of data corresponding to the outflow.

        Args:
            fig_height (float, optional): Figure height. Defaults to 2.0.
            ax (matplotlib Axes object, optional): Axis for plotting. Default is None (new axis is defined).
            feature_coords (list of SkyCoord objects, optional): Feature coordinates to plot. Default is None.
            plot_title (bool, optional): If True, plots the given text. Default is False.
            cmap (str, optional): Matplotlib colormap. Default is "magma".
            show (bool, optional): If True, includes plt.show(). Default is True.
            save_as (str, optional): If provided, the image is saved in a file with this name. Default is None.
            dpi (int, optional): DPI of figure (if saved)
            stretch (str, optional): Stretch for normalization. Default is "sqrt".
            percent (float, optional): Percentile value used to determine the pixel value of maximum cut level for plotting. Default is 99.0.
            show_xlabels (bool, optional): If True, x-axis labels will be plotted
            show_ylabels (bool, optional): If True, y-axis labels will be plotted
        Returns:
        """

        figsize = (
            fig_height * self.window_x_arcsec / self.window_y_arcsec,
            fig_height,
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = ax
            fig = ax.figure

        fig.subplots_adjust(bottom=0.25)

        # Set NaNs to 0
        data = np.nan_to_num(self.window_hdu.data, copy=True, nan=0.0)

        # Plot the image
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            norm=simple_norm(
                data,
                stretch=stretch,
                percent=percent,
                max_percent=percent,
            ),
        )

        # Label axes based on offset from the center (in arcseconds)
        pixel_scale = ImageUtilities.get_pixel_scale(self.window_hdu)

        # Image size in arcseconds
        window_x_arcsec = np.shape(data)[1] * pixel_scale
        window_y_arcsec = np.shape(data)[0] * pixel_scale

        if show_xlabels:
            # Ticks and labels
            x_ticks = window_x_arcsec * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            x_positions = x_ticks / pixel_scale
            x_labels = [str(np.round(x - (window_x_arcsec / 2), 1)) for x in x_ticks]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Offset [arcsec]")
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        if show_ylabels:
            y_ticks = window_y_arcsec * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            y_positions = y_ticks / pixel_scale
            y_labels = [str(np.round((y - (window_y_arcsec / 2)), 1)) for y in y_ticks]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
            ax.set_ylabel("Offset [arcsec]")
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Plot filter name
        if plot_title:
            ax.text(
                0.85, 0.8, plot_title, color="white", transform=ax.transAxes
            )  # the transform changes the text location to fraction of the subplot size

        # If feature coordinates are provided, plot them
        if feature_coords:

            feature_coords_pix = [
                self.window_wcs.world_to_pixel(feature_coord)
                for feature_coord in feature_coords
            ]

            for pix_coord in feature_coords_pix:
                if ImageUtilities.source_in_image(
                    self.window_hdu.data.shape, pix_coord
                ):
                    plt.scatter(
                        pix_coord[0],
                        pix_coord[1],
                        color="red",
                        # facecolor="none",
                        marker="x",
                        s=10,
                    )
        # If patches (like rectangles, etc.) are provided, plot them
        if len(patches) > 0:
            [ax.add_patch(patch) for patch in patches]

        if save_as:
            plt.tight_layout()
            plt.savefig(save_as, dpi=dpi, transparent=True)

        if show:
            plt.show(fig)

        # Label axes based on offset from the center (in arcseconds)
        pixel_scale = ImageUtilities.get_pixel_scale(self.window_hdu)

    def _outflow_region(self, image_hdu, rotation_angle):
        """Sets the window_hdu attribute to an hdu of the data within a certain window around the outflow
        Args:
        image_hdu (FITS ImageHDU): HDU that contains the image data
        rotation_angle (float): Angle (in degrees) by which the image needs to be rotated to make the outflow horizontal
        """

        # Perform an initial crop before rotating
        diagonal_arcsec = np.hypot(self.window_x_arcsec, self.window_y_arcsec)
        precrop_size_arcsec = diagonal_arcsec * 1.2
        hdu_cut_initial = ImageUtilities.cutout(
            image_hdu,
            self.central_coord,
            precrop_size_arcsec,
            precrop_size_arcsec,
        )

        # Rotate the image
        hdu_rotated = ImageUtilities.rotate_hdu(
            hdu_cut_initial, rotation_angle_deg=rotation_angle
        )

        # Cut to the final size
        hdu_cut = ImageUtilities.cutout(
            hdu_rotated,
            self.central_coord,
            self.window_x_arcsec,
            self.window_y_arcsec,
        )

        # Set the window_hdu attribute to the new HDU
        self.window_hdu = hdu_cut

        return hdu_cut

    def compute_dendrogram(
        self,
        sigma_clip_val=3,
        min_value_factor=3.0,
        min_delta_factor=1.0,
        min_npix=100,
        mask_bright=False,
        noise=None,
    ):
        """Computes dendrogram from the outflow data.

        Args:
            sigma_clip_val (float, optional): Sigma value for sigma clipping. Defaults to 3.
            min_value_factor (float, optional): The minimum value (how many times the noise) to consider in the dendrogram. Defaults to 3.
            min_delta_factor (float, optional): How significant (how many times the noise) a leaf has to be in order to be considered an independent entity. Defaults to 1.
            min_npix (int, optional): Minimum number of pixels/values needed for a leaf to be considered an independent entity. Defaults to 50.
            mask_bright (bool, optional): If True, masks out amy structures with NaNs or a maximum value > 1000 x the noise using a circular aperture. Defaults to False.
            save_as (str, optional): If provided, the dendrogram is saved in a file with this name. Accepted formats are FITS (extension ".fits") and HDF5 (extension ".hdf5"). Default is None.

        Returns:
            dend(astrodendro.dendrogram.Dendrogram): Dendrogram of structures meeting the criteria
        """
        data = self.window_hdu.data

        # If noise is not provided, use the window HDU to calculate it
        if noise == None:
            noise = StdBackgroundRMS(
                sigma_clip=SigmaClip(sigma=sigma_clip_val)
            ).calc_background_rms(data)

        dend = Dendrogram.compute(
            data,
            min_value=min_value_factor * noise,
            min_delta=min_delta_factor * noise,
            min_npix=min_npix,
        )

        if mask_bright == True:
            # Mask out any structures with NaNs or a maximum value > 1000 x the noise
            mask = np.zeros(np.shape(data))
            for structure in dend.leaves:

                #structrure_mask = data * structure.get_mask()

                remove_src = (
                    np.any(np.isnan(structure.values()))
                    or structure.vmax >= 1000 * noise
                )
                if remove_src == True:
                    stat = PPStatistic(structure)
                    x, y, r = (
                        stat.x_cen.value,
                        stat.y_cen.value,
                        stat.major_sigma.value * 3,
                    )
                    mask_ap = (
                        CircularAperture([x, y], r)
                        .to_mask()
                        .to_image(shape=data.shape)
                        .astype(bool)
                    )
                    mask += mask_ap  # structrure_mask

            # Re-compute dendrogram w/ masked data
            dend = Dendrogram.compute(
                data * ~mask.astype(bool),
                min_value=min_value_factor * noise,
                min_delta=min_delta_factor * noise,
                min_npix=min_npix,
            )

        self._dendrogram = dend

        return dend

    def _extract_structure_data(self, structures):
        """Extract data from structure objects"""
        structure_data = []
        for s in structures:
            data = {
                "idx": s.idx,
                "mask": s.get_mask() if hasattr(s, "get_mask") else None,
                "level": getattr(s, "level", None),
                "is_leaf": s.is_leaf if hasattr(s, "is_leaf") else None,
            }
            structure_data.append(data)
        return structure_data

    @staticmethod
    def _cropped_structure_mask(structure, shape, pad=1):
        """ Builds a cropped mask based on the structure's bounding box, instead
        of creating a mask the size of the entire image (speeds up contour plotting)

        Args:
            structure (astrodendro structure): Structure to build the mask for
            shape (tuple): Shape (ny, nx) of the full image the structure is in
            pad (int, optional): Padding (in pixels) around the structure's bounding
                box. At least 1 px of padding is needed so the contour ends at the
                edge of the crop. Defaults to 1.

        Returns:
            cropped_mask (np.ndarray): Local mask array, cropped to the padded
                bounding box.
            x0, y0 (int): Pixel offset of the cropped mask's origin within the full
                image (i.e. cropped_mask[0, 0] corresponds to full-image pixel
                (y0, x0)).
        """
        y_idx, x_idx = structure.indices(subtree=True)
        ny, nx = shape

        y0 = max(int(y_idx.min()) - pad, 0)
        y1 = min(int(y_idx.max()) + pad + 1, ny)
        x0 = max(int(x_idx.min()) - pad, 0)
        x1 = min(int(x_idx.max()) + pad + 1, nx)

        cropped_mask = np.zeros((y1 - y0, x1 - x0), dtype=int)
        cropped_mask[y_idx - y0, x_idx - x0] = 1

        return cropped_mask, x0, y0

    def select_structures(
        self,
        dendrogram,
        ax=None,
        ignore_leaves=False,
        stretch="sqrt",
        figsize=(10, 4),
        percent=None,
    ):
        """Enables the interactive selection of dendrogram structures

        Args:
            dendrogram (astrodendro.dendrogram.Dendrogram): Dendrogram of structures
            ax (matplotlib Axes object, optional): Axis for plotting. Default is None (new axis is defined).
            ignore_leaves (bool, optional): If True, ignores independent leaves not connected to any other structures. Defaults to False.
            stretch (str, optional): Stretch for normalization. Default is "sqrt".
            figsize (tuple, optional): _description_. Defaults to (10, 4).
            percent (floar, optional): Percentile value used to determine the pixel value of maximum cut level for plotting. Default is 99.0.
            cmap (str, optional): Matplotlib colormap. Default is "magma".
            show (bool, optional): If True, includes plt.show(). Default is True.
            save_as (str, optional): If provided, the image is saved in a file with this name. Default is None.
            stretch (str, optional): Stretch for normalization. Default is "sqrt".
            percent (floar, optional): Percentile value used to determine the pixel value of maximum cut level for plotting. Default is 99.0.

        Returns:
            _type_: _description_
        """
        # Initialize selection tracking
        selected_structures = []
        structure_contours = {}
        structure_patches = {}
        level = 0

        # Plot outflow
        fig, ax = plt.subplots(figsize=figsize)

        self.plot_outflow(
            ax=ax,
            stretch=stretch,
            percent=percent if percent else 99,
            show=False,
            save_as=None,
        )

        # Make room for the buttons
        fig.subplots_adjust(bottom=0.35)

        def plot_contours(structure_level):
            """Plot contours for all structures and store references.
            Args:
                structure_level (int): Dendrogram level of structures to plot
            """

            # Indicate that these variables will be modified in select_structures()
            nonlocal selected_structures, structure_contours, structure_patches
            selected_structures = []
            structure_contours = {}
            structure_patches = {}
            structures = []

            # Iterate through structures and store ones we want to plot
            for s in dendrogram.all_structures:
                struct_level = getattr(s, "level", 0)
                # If the structure is at the indicated level, store it for plotting
                if struct_level == structure_level:
                    structures.append(s)
                # If structure is BELOW the indicated level but is the highest-level structure in its branch, store it
                elif struct_level < level and len(s.children) == 0:
                    structures.append(s)

            for structure in structures:
                # If ignore_leaves is True, this structure will be ignored if it is an independent leaf
                if ignore_leaves:
                    leaf_condition = structure.is_leaf and structure.parent == None
                else:
                    leaf_condition = False

                if hasattr(structure, "get_mask") and not leaf_condition:
                    # Plot the contour using a cropped mask
                    mask, x0, y0 = self._cropped_structure_mask(
                        structure, self.window_hdu.data.shape
                    )
                    contour_set = ax.contour(
                        mask,
                        levels=[0.5],
                        colors=["white"],
                        alpha=0.9,
                        linewidths=1,
                        extent=(x0, x0 + mask.shape[1] - 1, y0, y0 + mask.shape[0] - 1),
                    )
                    # Store the contour paths for click detection
                    structure_contours[structure.idx] = {
                        "structure": structure,
                        "contour_set": contour_set,
                        "paths": [],
                        "mask": mask,
                        "origin": (x0, y0),
                    }

                    # Extract paths from contour collections
                    try:
                        collections = getattr(contour_set, "collections", [])
                        for collection in collections:
                            paths = getattr(collection, "_paths", [])
                            if not paths:
                                try:
                                    paths = collection.get_paths()
                                except AttributeError:
                                    continue
                            for path in paths:
                                structure_contours[structure.idx]["paths"].append(path)
                    except (AttributeError, TypeError):
                        pass

        def on_click(event):
            """Handle click events"""

            if event.inaxes != ax:
                return

            if event.button == 1:  # Left click
                clicked_point = (event.xdata, event.ydata)
                structure = find_clicked_structure(clicked_point)

                if structure:
                    if structure.idx in selected_structures:
                        # If already selected, remove from selection
                        remove_structure(structure)
                    else:
                        # Otherwise, add to selection
                        add_structure(structure)

                        update_display()

        def find_clicked_structure(point):
            """Finds which structure was clicked based on contour paths or mask"""
            x, y = point

            for struct_id, contour_data in structure_contours.items():
                structure = contour_data["structure"]

                # First try path-based detection if paths are available
                if contour_data["paths"]:
                    for path in contour_data["paths"]:
                        try:
                            if path.contains_point((x, y)):
                                return structure
                        except (AttributeError, TypeError):
                            continue

                # Fallback to mask-based detection (mask is cropped to the
                # structure's bounding box, so offset the click by its origin)
                mask = contour_data["mask"]
                x0, y0 = contour_data["origin"]
                try:
                    # Convert coordinates to integer indices local to the crop
                    ix, iy = int(round(x)) - x0, int(round(y)) - y0
                    if (
                        0 <= ix < mask.shape[1]
                        and 0 <= iy < mask.shape[0]
                        and mask[iy, ix]
                    ):
                        return structure
                except (ValueError, IndexError, TypeError):
                    continue

            return None

        def add_structure(structure):
            """Add structure to selection and highlight it"""
            if structure.idx not in selected_structures:
                selected_structures.append(structure.idx)

                # Reuse the cropped mask already computed in plot_contours
                contour_data = structure_contours[structure.idx]
                mask = contour_data["mask"]
                x0, y0 = contour_data["origin"]
                highlight_contour = ax.contour(
                    mask,
                    levels=[0.5],
                    colors=["lime"],
                    alpha=0.9,
                    linewidths=3,
                    extent=(x0, x0 + mask.shape[1] - 1, y0, y0 + mask.shape[0] - 1),
                )

                structure_patches[structure.idx] = highlight_contour

        def remove_structure(structure):
            """Remove structure from selection and remove highlight."""
            if structure.idx in selected_structures:
                selected_structures.remove(structure.idx)

                # Remove highlight
                if structure.idx in structure_patches:
                    contour_set = structure_patches[structure.idx]
                    remove_contour_set(contour_set)
                    del structure_patches[structure.idx]

        def remove_contour_set(contour_set):
            """Safely remove a contour set from the plot."""
            try:
                # Try different ways to remove the contour
                if hasattr(contour_set, "collections"):
                    collections = getattr(contour_set, "collections", [])
                    for collection in collections:
                        if collection in ax.collections:
                            collection.remove()
                elif hasattr(contour_set, "remove"):
                    contour_set.remove()
                else:
                    # Manual removal from axes collections
                    for collection in list(ax.collections):
                        if collection in getattr(contour_set, "collections", []):
                            collection.remove()
            except (AttributeError, ValueError):
                # If removal fails, try to clear and redraw
                pass

        def update_display():
            """Update the display with current selection info."""
            info_text.set_text(get_info_text())
            fig.canvas.draw_idle()

        def update_level(val):
            "Update the current level."
            nonlocal level
            # Remove highlight
            for collection in list(ax.collections):
                collection.remove()

            level = self._level_slider.val
            # Plot all contours and store them
            plot_contours(level)
            update_display()

        def get_info_text():
            """Generate info text showing instructions and selected structures."""
            text = "Click on contours to select/deselect structures\n"
            text += f"Selected structures: {len(selected_structures)}\n"
            if selected_structures:
                text += f"IDs: {sorted(selected_structures)}"
            return text

        def get_selected_structures():
            """Return list of selected structure objects."""
            selected = []
            for struct_id in selected_structures:
                for structure in dendrogram.all_structures:
                    if structure.idx == struct_id:
                        selected.append(structure)
                        break
            return selected

        def clear_selection(sevent):
            """Clear all selected structures."""
            # Remove all highlights
            for struct_id in list(structure_patches.keys()):
                contour_set = structure_patches[struct_id]
                remove_contour_set(contour_set)

            structure_patches.clear()
            selected_structures.clear()
            update_display()

        def finish_selecting(event):
            # Get final selected structures
            final_selection = get_selected_structures()
            self._selected_structures = final_selection
            self._selected_structure_data = self._extract_structure_data(
                final_selection
            )

            # Create a data mask corresponding to the selected structures
            for struct in final_selection:
                struct_mask = struct.get_mask()
                self._selected_structure_mask += struct_mask

            # Disable buttons to prevent further clicking
            self._finish_button.ax.set_visible(False)
            self._clear_button.ax.set_visible(False)
            # Set title to indicate selection is finished
            ax.set_title(
                "Selection complete",
                color="green",
                fontweight="bold",
            )

            # Disconnect event handlers to prevent further clicking
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.draw()
            # Turn interactive off
            # plt.ioff()

            return final_selection

        # Plot contours and store them
        plot_contours(level)
        # Connect the click event
        cid = fig.canvas.mpl_connect("button_press_event", on_click)

        # Add text to show instructions and selected structures
        info_text = fig.text(
            0.05,
            0.99,
            get_info_text(),
            transform=fig.transFigure,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Add control buttons
        ax_clear = plt.axes([0.34, 0.17, 0.2, 0.07])
        self._clear_button = Button(ax_clear, "Clear Selections")
        self._clear_button.on_clicked(clear_selection)
        ax_finish = plt.axes([0.55, 0.17, 0.2, 0.07])
        self._finish_button = Button(ax_finish, "Finish Selecting")
        self._finish_button.on_clicked(finish_selecting)

        # Add structure level slider
        axlevel = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self._level_slider = Slider(
            axlevel,
            label="Structure Level",
            valmin=0,
            valstep=1,
            valmax=np.max([s.level for s in dendrogram.all_structures]),
            valinit=0,
        )
        self._level_slider.on_changed(update_level)

        plt.show()

        # return self._selected_structures

    def get_selected_structure_data(self):
        """Get the extracted data from selected structures (survives pickling)."""
        return (
            self._selected_structure_data
            if hasattr(self, "_selected_structure_data")
            else []
        )

    def compute_PA(
        self,
        min_delta=5,
        save_as=None,
        figsize=(5, 3),
        dpi=300,
        ax=None,
        show=True,
        separate_lobes=False,
        fit_positions=None,
    ):

        if not self._selected_structures:
            raise ValueError("No structures selected. Run select_structures() first.")

        data = self.window_hdu.data
        header = self.window_hdu.header
        img_shape = np.shape(data)
        wcs = WCS(header)

        outflow_coord = self.central_coord
        outflow_xy = wcs.world_to_pixel(outflow_coord)

        mask = self._selected_structure_mask
        masked_img = data * mask
        # m = masked_img <= 100 * noise
        # masked_img *= m
        masked_img = np.nan_to_num(masked_img, nan=0)
        # Get the [x,y] positions of pixels above brightness threshold
        pix_xy = np.flip(np.asarray(np.where(masked_img > 0)).T)

        # Get the x-distance from each pixel to the outflow central coord
        pix_xdist = np.array([x - outflow_xy[0] for x, _ in pix_xy])

        # Get the world coordinates of these pixels
        # pix_coords = [wcs.pixel_to_world(x, y) for x, y in pix_xy]

        # Get the x-distance from each pixel to the outflow central coord
        pix_xdist = np.array([x - outflow_xy[0] for x, _ in pix_xy])

        # Get the world coordinates of these pixels
        # pix_coords = [wcs.pixel_to_world(x, y) for x, y in pix_xy]
        ra, dec = wcs.pixel_to_world_values(pix_xy[:, 0], pix_xy[:, 1])
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        outflow_ra_deg = outflow_coord.ra.to(u.deg).value
        outflow_ra_rad = outflow_coord.ra.to(u.rad).value
        outflow_dec_deg = outflow_coord.dec.to(u.deg).value
        outflow_dec_rad = outflow_coord.dec.to(u.rad).value

        d_ra = np.radians(ra - outflow_ra_deg)
        pa_rad = np.arctan2(
            np.sin(d_ra),
            np.cos(outflow_dec_rad) * np.tan(dec_rad)
            - np.sin(outflow_dec_rad) * np.cos(d_ra),
        )
        pix_pas = np.degrees(pa_rad) % 360  # deg

        # pix_coords_left = np.array(pix_coords)[pix_xdist < 0]
        # pix_coords_right = np.array(pix_coords)[pix_xdist > 0]

        # pix_pas = np.array(
        #     [
        #         coord.position_angle(outflow_coord).to(u.deg).value
        #         for coord in pix_coords
        #     ]
        # )

        # Make a "correction" array that has a value of -180 anywhere the x-distance defined above is negative
        pa_corr = np.array(pix_pas > 180) * -180
        #  pa_corr = np.array(pix_pas > 180) * -180
        pix_pas += pa_corr
        pix_pas_left = pix_pas[pix_xdist < 0]
        pix_pas_right = pix_pas[pix_xdist > 0]

        # Plot a histogram of the pixel PAs
        # if ax is None:
        #     fig, ax = plt.subplots()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = ax
            fig = ax.figure

        pixel_scale = ImageUtilities.get_pixel_scale(self.window_hdu)

        counts, bins, patches = ax.hist(
            pix_pas,
            weights=np.ones(len(pix_pas)) / len(pix_pas),
            bins=30,
            alpha=0.7,
            color="lightgrey",  # "skyblue",
            edgecolor="black",
            label="Pixel PAs",
        )
        # Calculate the bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Define a gaussian function
        def gaussian(x, amplitude, mean, std_dev):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))

        def bimodal(x, amplitude1, mean1, std_dev1, amplitude2, mean2, std_dev2):
            return gaussian(x, amplitude1, mean1, std_dev1) + gaussian(
                x, amplitude2, mean2, std_dev2
            )

        # Fit Gaussian to histogram data
        try:
            if separate_lobes == False:
                # Initial guess for parameters [amplitude, mean, std_dev]
                initial_guess = [
                    np.nanmax(pix_pas),
                    np.nanmean(pix_pas),
                    np.nanstd(pix_pas),
                ]
                popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
                amplitude_fit, mean_fit, std_fit = popt
                # Generate smooth curve for plotting
                x_fit = np.linspace(min(bins), max(bins), 200)
                y_fit = gaussian(x_fit, amplitude_fit, mean_fit, std_fit)

                # Plot the fitted curve
                ax.plot(
                    x_fit,
                    y_fit,
                    "-",
                    linewidth=2,
                    color="limegreen",
                    label=f"Gaussian Fit\nμ = {mean_fit:.2f}, σ = {std_fit:.2f}",
                )
            else:
                if fit_positions == None:
                    fit_positions = [
                        np.nanmedian(pix_pas_left),
                        np.nanmedian(pix_pas_right),
                    ]
                    #print(fit_positions)

                initial_guess = [
                    1,
                    fit_positions[0],
                    np.nanstd(pix_pas_left),
                    1,
                    fit_positions[1],
                    np.nanstd(pix_pas_right),
                ]
                #print(initial_guess)

                popt, pcov = curve_fit(
                    bimodal, bin_centers, counts, p0=initial_guess, bounds=([0, 180])
                )
                (
                    amplitude_fit1,
                    mean_fit1,
                    std_fit1,
                    amplitude_fit2,
                    mean_fit2,
                    std_fit2,
                ) = popt
                # Generate smooth curve for plotting
                x_fit = np.linspace(min(bins), max(bins), 200)
                y_fit = bimodal(
                    x_fit,
                    amplitude_fit1,
                    mean_fit1,
                    std_fit1,
                    amplitude_fit2,
                    mean_fit2,
                    std_fit2,
                )

                pa_left = mean_fit1
                pa_right = mean_fit2

                # Flip right lobe to point in the same direction as left lobe
                pa_right_flipped = pa_right + 180.0

                # Circular mean of the two directions
                angles_rad = np.deg2rad([pa_left, pa_right_flipped])
                mean_sin = np.mean(np.sin(angles_rad))
                mean_cos = np.mean(np.cos(angles_rad))
                mean_pa = np.rad2deg(np.arctan2(mean_sin, mean_cos))

                # Normalize to [0, 180)
                mean_pa = mean_pa % 180.0

                # Plot the fitted curve
                ax.plot(
                    x_fit,
                    y_fit,
                    "-",
                    color="limegreen",
                    linewidth=2,
                    label=f"μ$_1$ = {pa_left:.2f}, σ$_1$ = {std_fit1:.2f}, \nμ$_2$ = {pa_right:.2f}, σ$_2$ = {std_fit2:.2f}, \nμ = {mean_pa:.2f}",
                    # label=f"μ$_1$ = {mean_fit1:.2f}, σ$_1$ = {std_fit1:.2f}, \nμ$_2$ = {mean_fit2:.2f}, σ$_2$ = {std_fit2:.2f}, \nμ = {(mean_fit1+mean_fit2)/2:.2f}",
                )

        except RuntimeError as e:
            print(f"Fitting failed: {e}")

        # Formatting
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel(r"PA [$\deg$]")  # ,fontsize=16)
        # ax.set_ylabel('%')
        ax.legend()  # (fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_as:
            plt.savefig(save_as, dpi=dpi)

        if show:
            plt.show()

        if separate_lobes == False:
            try:
                self._position_angle = mean_fit
                self._position_angle_err = std_fit
                return mean_fit, std_fit, x_fit, y_fit, pix_pas
            except UnboundLocalError:  # fitting failed
                mean_pa = np.mean(pix_pas)
                self._position_angle = mean_pa
                self._position_angle_err = None
                print("Fit failed. Average pixel PA calculated instead.")
                return None
        else:
            try:
                self._position_angle = mean_pa  # (mean_fit1 + mean_fit2) / 2
                self._position_angle_err = np.sqrt(std_fit1**2 + std_fit2**2) / 2
                return mean_fit1, std_fit1, mean_fit2, std_fit2, x_fit, y_fit, pix_pas
            except UnboundLocalError:  # fitting failed
                # Estimate outflow PA from the PAs of all outflow pixels.
                # Treats angles as axial (0 == 180).
                # Double the angles to map axial data onto a full circle,
                # take the vector mean, then halve
                pixel_pas_deg = np.array(
                    [
                        coord.position_angle(outflow_coord).to(u.deg).value
                        for coord in pix_coords
                    ]
                )
                mean_pa = np.mean(pixel_pas_deg)
                self._position_angle = mean_pa
                self._position_angle_err = None
                print("Fit failed. Average pixel PA calculated instead.")
                return None

    def structure_plots(self, savename=None, figsize=(5, 4), save_as=None, dpi=300):

        fig = plt.figure(figsize=figsize)
        fig.canvas.header_visible = False  # Gets rid of "Figure 1" on top

        ax = plt.subplot(2, 1, 1)

        # Outflow plot with highlighted structures
        self.plot_outflow(ax=ax, show=False)
        dendrogram = self._dendrogram
        p = dendrogram.plotter()
        structure_inds = [s.idx for s in self._selected_structures]
        [
            p.plot_contour(ax, structure=branch_ind, lw=3, colors="limegreen")
            for branch_ind in structure_inds
        ]

        # Dendrogram plot with highlighted structures
        ax2 = fig.add_subplot(2, 1, 2)
        p.plot_tree(ax2, color="k")

        [
            p.plot_tree(ax2, structure=branch_ind, color="limegreen")
            for branch_ind in structure_inds
        ]

        ax2.semilogy()
        ax2.set_xlabel("Structure #", fontsize=11)
        ax2.set_ylabel("MJy/sr", fontsize=11)

        plt.tight_layout()

        if save_as:
            plt.savefig(save_as, dpi=dpi)

        plt.show()

    def get_box(self, hdu):
        """Creates a matplotlib box patch corresponding to the outflow window in a larger image

        Args:
            hdu (FITS Image HDU): HDU of image the box will be plotted on
        Returns:
            box (matplotlib.patches.Polygon): box patch corresponding to the outflow on the provided image
        """
        wcs = WCS(hdu.header)
        central_pix_coord = wcs.world_to_pixel(self.central_coord)
        pixel_scale = ImageUtilities.get_pixel_scale(hdu)
        width = self.window_x_arcsec / pixel_scale
        height = self.window_y_arcsec / pixel_scale
        image_pa = hdu.header["PA_APER"]

        # If the PA has been calculated, use that as the box rotation angle
        if self._position_angle:
            rotation_angle = self._position_angle
        # Otherwise, use the image rotation angle
        else:
            rotation_angle = self._img_rotation_angle
        # Create unrotated corners relative to center
        corners = np.array(
            [
                [-width / 2, -height / 2],
                [width / 2, -height / 2],
                [width / 2, height / 2],
                [-width / 2, height / 2],
            ]
        )

        # Rotation matrix
        theta = np.radians(rotation_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate corners
        rotated_corners = corners @ rotation_matrix.T

        # Translate to actual position
        vertices = rotated_corners + np.array(
            [central_pix_coord[0], central_pix_coord[1]]
        )

        box = Polygon(
            vertices, fill=False, edgecolor="red", linewidth=1, linestyle="--"
        )

        return box

    def get_arrow(self, hdu, color="white"):
        """Create an arrow patch corresponding to the outflow PA and length

            TODO: Fix this so that the arrow spans the outflow (need to translate x-pos in old)
        Args:
            hdu (_type_): _description_
            color (str, optional): Arrow color. Defaults to "red".

        Returns:
            arrow (matplotlib.patches.FancyArrowPatch): arrow patch corresponding to the outflow in a given image
        """
        wcs = WCS(hdu.header)
        central_pix_coord = wcs.world_to_pixel(self.central_coord)

        if not self.selected_structures and not self._position_angle:
            raise ValueError(
                "No position angle stored. Run select_structures() and compute_PA() first."
            )
            return None
        elif not self._position_angle:
            raise ValueError("No position angle stored. Run compute_PA() first.")
            return None

        # Get the arrow direction from the hdu's WCS (PA_APER isn't updated when
        # the image is rotated, so it's wrong for the rotated outflow window)
        offset_coord = self.central_coord.directional_offset_by(
            self._position_angle * u.deg, 1 * u.arcsec
        )
        offset_pix_coord = wcs.world_to_pixel(offset_coord)
        rotation_angle = np.degrees(
            np.arctan2(
                offset_pix_coord[1] - central_pix_coord[1],
                offset_pix_coord[0] - central_pix_coord[0],
            )
        )
        pixel_scale = ImageUtilities.get_pixel_scale(hdu)

        # # Find the max and min x values in the selected structures and set the arrow length to their difference
        masked_inds = np.where(self._selected_structure_mask == 1)
        length = np.max(masked_inds[1]) - np.min(masked_inds[1])
        #(length)
        # length = self.window_x_arcsec / pixel_scale
        # Set the height to the window height
        height = self.window_y_arcsec / pixel_scale

        # Create unrotated corners relative to center
        ends = np.array(
            [
                [-length / 2, 0],
                [length / 2, 0],
            ]
        )

        # Rotation matrix
        theta = np.radians(rotation_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate corners
        rotated_corners = ends @ rotation_matrix.T

        # Translate to actual position
        vertices = rotated_corners + np.array(
            [central_pix_coord[0], central_pix_coord[1]]
        )

        arrow = FancyArrowPatch(
            vertices[0],
            vertices[1],
            color=color,
            arrowstyle="<->, head_length=2, head_width=2",
            linestyle=":",
            linewidth=1,
        )

        return arrow

    def __getstate__(self):
        """
        Remove all unpicklable objects before saving.
        """
        state = self.__dict__.copy()

        # Remove unpicklable matplotlib widgets
        for widget_attr in ["_clear_button", "_finish_button", "_level_slider"]:
            state.pop(widget_attr, None)

        # Remove unpicklable structures and dendrogram
        state.pop("_selected_structures", None)
        state.pop("_dendrogram", None)

        # Remove HDU  (will be saved separately)
        state.pop("window_hdu", None)

        # Convert SkyCoord to simple format
        if "central_coord" in state and state["central_coord"] is not None:
            try:
                coord = state["central_coord"]
                state["central_coord"] = {
                    "_pickled_coord": True,
                    "ra": float(coord.ra.deg),
                    "dec": float(coord.dec.deg),
                    "frame": str(coord.frame.name),
                }
            except Exception as e:
                print(f"Warning: Could not serialize central_coord: {e}")
                state.pop("central_coord", None)

        # Clean up structure data - ensure everything is plain types
        if "_selected_structure_data" in state and state["_selected_structure_data"]:
            cleaned_data = []
            for item in state["_selected_structure_data"]:
                cleaned_item = {}
                for key, value in item.items():
                    if isinstance(value, np.ndarray):
                        cleaned_item[key] = np.array(value)  # Make a clean copy
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_item[key] = value.item()  # Convert to Python type
                    else:
                        cleaned_item[key] = value
                cleaned_data.append(cleaned_item)
            state["_selected_structure_data"] = cleaned_data

        return state

    def __setstate__(self, state):
        """Restore object after loading."""
        self.__dict__.update(state)

        # Reconstruct SkyCoord
        if "central_coord" in state and isinstance(state["central_coord"], dict):
            if state["central_coord"].get("_pickled_coord"):
                coord_data = state["central_coord"]
                self.central_coord = SkyCoord(
                    ra=coord_data["ra"],
                    dec=coord_data["dec"],
                    unit="deg",
                    frame=coord_data["frame"],
                )

        # Initialize unpickled attributes
        self._selected_structures = []
        self._dendrogram = None
        self.window_hdu = None

    def save(self, dir_name):
        """
        Save the Outflow instance and HDU.

        Parameters:
        -----------
        dir_name : str
            Base directory name (without extension)
        """
        # Remove extension if provided
        dir_name = dir_name.replace(".pkl", "")

        print(os.path.exists(dir_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Save HDU as FITS file
        if self.window_hdu is not None:
            hdu_filename = dir_name + "/outflow_hdu.fits"
            try:
                # Create a completely clean copy
                clean_hdu = fits.PrimaryHDU(
                    data=(
                        np.array(self.window_hdu.data)
                        if self.window_hdu.data is not None
                        else None
                    ),
                    header=fits.Header(self.window_hdu.header.cards),
                )
                clean_hdu.writeto(hdu_filename, overwrite=True)
                print(f"HDU saved to {hdu_filename}")
            except Exception as e:
                print(f"Warning: Could not save HDU: {e}")

        # Save the outflow
        pkl_filename = dir_name + "/outflow.pkl"
        try:
            with open(pkl_filename, "wb") as f:
                pickle.dump(self, f)
            print(f"Outflow saved to {pkl_filename}")
        except Exception as e:
            print(f"Error saving outflow: {e}")

        # Save the dendrogram
        dendro_filename = dir_name + "/dendrogram.fits"
        self._dendrogram.save_to(filename=dendro_filename)

    def reconstruct_structures(self, dendrogram):
        """
        Reconstruct structure objects from saved indices.
        Requires dendrogram to be loaded.

        Returns:
        --------
        list
            List of structure objects
        """
        structures = []
        for data in self._selected_structure_data:
            idx = data["idx"]
            for s in dendrogram.all_structures:
                if s.idx == idx:
                    structures.append(s)
                    break

        self._selected_structures = structures
        return structures

    @classmethod
    def load(cls, dirname):
        """
        Load an Outflow instance from file.

        Parameters:
        -----------
        filename : str
            Base filename (with or without .pkl extension)
        """
        # Remove extension if provided
        base_filename = dirname.replace(".pkl", "")

        pkl_filename = dirname + "/outflow.pkl"
        if not os.path.exists(pkl_filename):
            raise FileNotFoundError(f"File {pkl_filename} not found")

        # Load the outflow
        with open(pkl_filename, "rb") as f:
            outflow = pickle.load(f)
        print(f"Outflow loaded from {pkl_filename}")

        dend_filename = dirname + "/dendrogram.fits"
        dend = Dendrogram.load_from(filename=dend_filename)
        outflow._dendrogram = dend
        outflow._selected_structures = outflow.reconstruct_structures(dend)

        # Load HDU
        hdu_filename = dirname + "/outflow_hdu.fits"
        if os.path.exists(hdu_filename):
            try:
                with fits.open(hdu_filename) as hdul:
                    outflow.window_hdu = fits.ImageHDU(
                        data=hdul[0].data.copy(), header=hdul[0].header.copy()
                    )
                print(f"HDU loaded from {hdu_filename}")
            except Exception as e:
                print(f"Warning: Could not load HDU: {e}")
        else:
            print(f"Warning: HDU file {hdu_filename} not found")

        return outflow

    @property
    def selected_structures(self):
        """Structures selected in the last interactive selection."""
        return self._selected_structures

    @property
    def position_angle(self):
        """
        Position angle of the outflow in degrees.

        Returns None if not yet calculated. Run calculate_position_angle() first.
        """
        return self._position_angle

    @property
    def position_angle_err(self):
        """
        Error in position angle in degrees.

        Returns None if not yet calculated. Run calculate_position_angle() first.
        """
        return self._position_angle_err

    @property
    def dendrogram(self):
        """
        Dendrogram

        Returns None if not yet calculated. Run compute_dendrogram() first.
        """
        return self._dendrogram


class RegionUtilities:
    @staticmethod
    def region_to_params(region_path, image_hdu):
        """Takes a file with DS9 regions and returns a dataframe of parameters corresponding to those regions
        Args:
            region_path (str): Path to region file
            image_hdu (FITS image HDU): HDU of image used to define the regions
        Returns:
            params_df (Pandas DataFrame): Dataframe of region parameters
        """
        wcs = WCS(image_hdu.header)
        pixel_scale = ImageUtilities.get_pixel_scale(image_hdu)
        sky_coords = []
        box_x_arcsec = []
        box_y_arcsec = []
        rotation_deg = []

        with open(region_path) as file:
            # Read in lines with box info and split up using ',' delimeter
            lines = [
                line.split("(")[1].split(")")[0].split(",")
                for line in file.read().splitlines()
                if line.startswith("box")
            ]
            # Access region parameters

            for line in lines:
                x, y = float(line[0]), float(line[1])
                sky_coords.append(wcs.pixel_to_world(x, y))

                # If y > x, swap
                if float(line[3]) > float(line[2]):
                    box_x_arcsec.append(float(line[3]) * pixel_scale)
                    box_y_arcsec.append(float(line[2]) * pixel_scale)
                    rotation_deg.append(float(line[4]) - 90)
                else:
                    box_x_arcsec.append(float(line[2]) * pixel_scale)
                    box_y_arcsec.append(float(line[3]) * pixel_scale)
                    rotation_deg.append(float(line[4]))

            params_df = df(
                {
                    "guess_coords": sky_coords,
                    "box_x_arcsec": box_x_arcsec,
                    "box_y_arcsec": box_y_arcsec,
                    "rotation_deg": rotation_deg,
                }
            )
            return params_df

    @staticmethod
    def region_to_outflow(image, region, central_coord):
        hdu = image["SCI", 1]
        data = hdu.data
        header = hdu.header
        wcs = WCS(header)
        pscale = ImageUtilities.get_pixel_scale(hdu)

        width = region["width"]
        height = region["height"]

        window_x_arcsec = np.max([width, height]) * pscale
        window_y_arcsec = np.min([width, height]) * pscale
        rotation = region["rotation"]

        if width < height:
            rotation -= 90

        outflow = Outflow(
            image,
            central_coord,
            position_angle=None,
            position_angle_err=None,
            window_y_arcsec=window_y_arcsec,
            window_x_arcsec=window_x_arcsec,
            rotation_angle=rotation,
        )
        return outflow

    @staticmethod
    def coords_to_catalog(coords):
        """
        Given a set of coordinates, creates a SkyCoord catalog

        Args:
            coords (list of astropy.coordinates.SkyCoord statement)

        Returns:
            catalog (SkyCoord statement): Catalog of sources
        """
        ra_arr = np.array([coord.transform_to("icrs").ra.value for coord in coords])
        dec_arr = np.array([coord.transform_to("icrs").dec.value for coord in coords])
        catalog = SkyCoord(ra=ra_arr * u.degree, dec=dec_arr * u.degree)
        return catalog

    @staticmethod
    def cross_match(initial_guess_coords, known_source_coords, tolerance_arcsec=5):
        """_summary_
        Args:
            initial_guess_coords (astropy.coordinates.SkyCoord statement): Coordinates of initial driving source position guess
            known_source_coords (list of astropy.coordinates.SkyCoord statements): Coordinates of known sources for cross matching
            tolerance_arcsec (float, optional): Maximum distance (in arcseconds) from the initial guess for the closest source to be considered a match
        Returns:
            SkyCoord statement: Closest source (or initial guess if none within tolerance)
            int: index of match in catalog (or None if none found)
            Quantity: Distance to closest source in arcseconds (or None if none found)
        """

        # Convert coord lists to catalogs
        known_source_catalog = RegionUtilities.coords_to_catalog(known_source_coords)

        tolerance_arcsec *= u.arcsec
        matched_coords = []

        idx, d2d, d3d = initial_guess_coords.match_to_catalog_sky(known_source_catalog)
        dist = d2d.to(u.arcsec)

        # If closest source in match catalog is less than tolerance, return
        if dist <= tolerance_arcsec:
            print(
                "Distance to closest source is {} (< {} arcsec tolerance).".format(
                    np.round(dist[0], 2), tolerance_arcsec
                )
            )
            return known_source_catalog[idx], idx, dist
        else:
            print(
                "No matches found (None returned). Distance to closest source is {} (> {} arcsec tolerance).".format(
                    np.round(dist[0], 2), tolerance_arcsec
                )
            )
            return initial_guess_coords, None, None


class RegionSelector:
    def __init__(self, image_data, title="JWST NIRCam Image"):
        """
        Interactive region selector for JWST images with rotation capability.

        Parameters:
        -----------
        image_data : 2D numpy array
            The image data to display
        title : str
            Title for the plot
        """
        self.image_data = image_data
        self.regions = []  # Stored regions
        self.current_rect = None
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.rotation_angle = 0  # Current rotation angle in degrees
        self.drawing_enabled = True  # Toggle for drawing mode

        # Calculate initial vmin and vmax
        valid_data = image_data[np.isfinite(image_data)]
        self.vmin = np.percentile(valid_data, 1)
        self.vmax_initial = np.percentile(valid_data, 99)
        self.vmax = self.vmax_initial
        self.vmax_min = np.percentile(valid_data, 90)
        self.vmax_max = np.percentile(valid_data, 99.9)

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2, left=0.1, top=0.93)

        # Display image with log scaling for better visualization
        self.im = self.ax.imshow(
            image_data,
            origin="lower",
            cmap="gray",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="nearest",
        )
        self.ax.set_title(title)
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        plt.colorbar(self.im, ax=self.ax, label="Flux")

        # Add toolbar reference to check mode
        self.toolbar = self.fig.canvas.toolbar

        # Create vmax slider at the top
        ax_vmax = plt.axes([0.2, 0.96, 0.6, 0.02])
        self.vmax_slider = Slider(
            ax_vmax,
            "vmax",
            self.vmax_min,
            self.vmax_max,
            valinit=self.vmax_initial,
            valfmt="%.1f",
        )
        self.vmax_slider.on_changed(self.update_vmax)

        # Create rotation slider
        ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
        self.rotation_slider = Slider(
            ax_slider, "Rotation (°)", -180, 180, valinit=0, valstep=1
        )
        self.rotation_slider.on_changed(self.update_rotation)

        # Create buttons
        ax_store = plt.axes([0.15, 0.05, 0.12, 0.04])
        ax_clear = plt.axes([0.3, 0.05, 0.12, 0.04])
        ax_finish = plt.axes([0.45, 0.05, 0.12, 0.04])
        ax_toggle = plt.axes([0.6, 0.05, 0.15, 0.04])

        self.btn_store = Button(ax_store, "Store Region")
        self.btn_clear = Button(ax_clear, "Clear Current")
        self.btn_finish = Button(ax_finish, "Finish")
        self.btn_toggle = Button(ax_toggle, "Drawing: ON")

        # Connect button callbacks
        self.btn_store.on_clicked(self.store_region)
        self.btn_clear.on_clicked(self.clear_current)
        self.btn_finish.on_clicked(self.finish_selection)
        self.btn_toggle.on_clicked(self.toggle_drawing)

        # Connect mouse events
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

        # Status text
        self.status_text = self.fig.text(
            0.5,
            0.01,
            "Click and drag to select region. Use toolbar to zoom/pan.",
            ha="center",
            fontsize=10,
        )

    def is_toolbar_active(self):
        """Check if a toolbar mode (zoom/pan) is active."""
        if self.toolbar is None:
            return False
        return self.toolbar.mode != ""

    def toggle_drawing(self, event):
        """Toggle drawing mode on/off."""
        self.drawing_enabled = not self.drawing_enabled
        if self.drawing_enabled:
            self.btn_toggle.label.set_text("Drawing: ON")
            self.status_text.set_text(
                "Drawing enabled. Click and drag to select region."
            )
        else:
            self.btn_toggle.label.set_text("Drawing: OFF")
            self.status_text.set_text("Drawing disabled. Use toolbar for zoom/pan.")
        self.fig.canvas.draw()

    def create_rotated_rectangle(self, x1, y1, x2, y2, angle):
        """
        Create vertices for a rotated rectangle.

        Parameters:
        -----------
        x1, y1 : float
            Starting corner coordinates
        x2, y2 : float
            Opposite corner coordinates
        angle : float
            Rotation angle in degrees

        Returns:
        --------
        vertices : array
            4x2 array of corner coordinates
        """
        # Center point
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Width and height
        width = x2 - x1
        height = y2 - y1

        # Create unrotated corners relative to center
        corners = np.array(
            [
                [-width / 2, -height / 2],
                [width / 2, -height / 2],
                [width / 2, height / 2],
                [-width / 2, height / 2],
            ]
        )

        # Rotation matrix
        theta = np.radians(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate corners
        rotated_corners = corners @ rotation_matrix.T

        # Translate to actual position
        vertices = rotated_corners + np.array([cx, cy])

        return vertices

    def on_press(self, event):
        """Handle mouse press event."""
        # Don't draw if toolbar is active or drawing is disabled
        if (
            event.inaxes != self.ax
            or self.is_toolbar_active()
            or not self.drawing_enabled
        ):
            return

        self.is_drawing = True
        self.start_point = (event.xdata, event.ydata)
        self.end_point = (event.xdata, event.ydata)

        # Create new rectangle (as polygon for rotation)
        if self.current_rect is not None:
            self.current_rect.remove()

        vertices = self.create_rotated_rectangle(
            self.start_point[0],
            self.start_point[1],
            self.end_point[0],
            self.end_point[1],
            self.rotation_angle,
        )

        self.current_rect = Polygon(
            vertices, fill=False, edgecolor="red", linewidth=2, linestyle="--"
        )
        self.ax.add_patch(self.current_rect)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """Handle mouse motion event."""
        if not self.is_drawing or event.inaxes != self.ax:
            return

        if self.current_rect is not None:
            # Update rectangle with current mouse position
            self.end_point = (event.xdata, event.ydata)
            vertices = self.create_rotated_rectangle(
                self.start_point[0],
                self.start_point[1],
                self.end_point[0],
                self.end_point[1],
                self.rotation_angle,
            )
            self.current_rect.set_xy(vertices)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse release event."""
        if not self.is_drawing:
            return

        self.is_drawing = False

        if event.inaxes != self.ax or self.start_point is None:
            return

        self.end_point = (event.xdata, event.ydata)

        # Update status
        self.status_text.set_text(
            f'Region drawn (rotation: {self.rotation_angle}°). Adjust slider or click "Store Region".'
        )
        self.fig.canvas.draw()

    def update_rotation(self, val):
        """Update the rotation angle of the current rectangle."""
        self.rotation_angle = val

        if (
            self.current_rect is not None
            and self.start_point is not None
            and self.end_point is not None
        ):
            vertices = self.create_rotated_rectangle(
                self.start_point[0],
                self.start_point[1],
                self.end_point[0],
                self.end_point[1],
                self.rotation_angle,
            )
            self.current_rect.set_xy(vertices)
            self.status_text.set_text(f"Rotation: {self.rotation_angle}°")
            self.fig.canvas.draw_idle()

    def update_vmax(self, val):
        """Update the vmax value for image display."""
        self.vmax = val
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
        self.fig.canvas.draw_idle()

    def store_region(self, event):
        """Store the current region."""
        if (
            self.current_rect is None
            or self.start_point is None
            or self.end_point is None
        ):
            self.status_text.set_text("No region to store. Draw a region first.")
            self.fig.canvas.draw()
            return

        # Get rectangle parameters
        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # Center point
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Width and height
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Get vertices
        vertices = self.create_rotated_rectangle(x1, y1, x2, y2, self.rotation_angle)

        # Store region
        region = {
            "center": (cx, cy),
            "width": width,
            "height": height,
            "rotation": self.rotation_angle,
            "vertices": vertices.copy(),
            "x_min": int(np.round(vertices[:, 0].min())),
            "x_max": int(np.round(vertices[:, 0].max())),
            "y_min": int(np.round(vertices[:, 1].min())),
            "y_max": int(np.round(vertices[:, 1].max())),
        }
        self.regions.append(region)

        # Change rectangle to stored style (solid green)
        self.current_rect.set_edgecolor("green")
        self.current_rect.set_linestyle("-")

        # Add region label
        label_y = vertices[:, 1].max() + 5
        self.ax.text(
            cx,
            label_y,
            f"R{len(self.regions)}",
            color="green",
            fontsize=10,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        self.current_rect = None
        self.start_point = None
        self.end_point = None
        self.rotation_slider.reset()
        self.rotation_angle = 0

        self.status_text.set_text(
            f'Region {len(self.regions)} stored. Draw another or click "Finish".'
        )
        self.fig.canvas.draw()

    def clear_current(self, event):
        """Clear the current selection."""
        if self.current_rect is not None:
            self.current_rect.remove()
            self.current_rect = None
            self.start_point = None
            self.end_point = None
            self.rotation_slider.reset()
            self.rotation_angle = 0
            self.status_text.set_text("Current selection cleared. Draw a new region.")
            self.fig.canvas.draw()
        else:
            self.status_text.set_text("No current selection to clear.")
            self.fig.canvas.draw()

    def finish_selection(self, event):
        """Finish selection and print results."""
        print(f"\n{'='*60}")
        print(f"Selection complete! {len(self.regions)} regions stored.")
        print(f"{'='*60}\n")

        for i, region in enumerate(self.regions, 1):
            print(f"Region {i}:")
            print(f"  Center: ({region['center'][0]:.1f}, {region['center'][1]:.1f})")
            print(f"  Size: {region['width']:.1f} × {region['height']:.1f} pixels")
            print(f"  Rotation: {region['rotation']:.1f}°")
            print(f"  Bounding box X: [{region['x_min']}, {region['x_max']}]")
            print(f"  Bounding box Y: [{region['y_min']}, {region['y_max']}]")
            print(f"  Vertices: ")
            for j, vertex in enumerate(region["vertices"], 1):
                print(f"    Corner {j}: ({vertex[0]:.1f}, {vertex[1]:.1f})")
            print()

        self.status_text.set_text(f"Finished! {len(self.regions)} regions selected.")
        self.fig.canvas.draw()

    def get_regions(self):
        """Return the list of stored regions."""
        return self.regions

    def show(self):
        """Display the interactive plot."""
        plt.show()
