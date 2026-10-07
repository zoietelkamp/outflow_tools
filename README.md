# outflow_tools

Tools for identifying protostellar outflow structures in astronomical images (e.g. JWST NIRCam, MIRI, ALMA) and estimating outflow position angles (PAs).

**Note: outflow_tools is in active development.**

Steps:

1. Cut out a window around an outflow's driving source and rotate it so the outflow is roughly horizontal.
2. Find emission structures in the window with [astrodendro](https://dendrograms.readthedocs.io/).
3. Select the structures that belong to the outflow with an interactive matplotlib widget.
4. Compute the PA of each selected pixel relative to the driving source, then fit a Gaussian (or two, for lobes with different orientations) to the distribution of pixel PAs.

## Installation

There is no package installer yet. Clone the repository and put `outflows.py` on your Python path (or work from inside the repo directory).

### Dependencies

- numpy, scipy, pandas, matplotlib
- astropy
- [astrodendro](https://github.com/dendrograms/astrodendro)
- [photutils](https://photutils.readthedocs.io/)
- [ipympl](https://matplotlib.org/ipympl/) (for the interactive widgets in Jupyter)

```bash
pip install numpy scipy pandas matplotlib astropy astrodendro photutils ipympl
```

The interactive structure selection needs an interactive matplotlib backend. In Jupyter, run `%matplotlib widget` first.

See example.ipynb for a tutorial for an example outflow.

## Module overview

The utilities are located in [outflows.py](outflows.py).

| Class | Purpose |
| --- | --- |
| `Outflow` | Main class. Builds the rotated cutout window, computes the dendrogram (`compute_dendrogram`), runs the interactive selection (`select_structures`), estimates the PA (`compute_PA`), plots (`plot_outflow`, `structure_plots`), and saves and loads (`save`, `load`). Results are available as the `position_angle`, `position_angle_err`, `selected_structures` and `dendrogram` properties |
| `ImageUtilities` | HDU helpers: reduce to 2D (`make_2d_hdu`), get the pixel scale, rotate an HDU, make cutouts with WCS information, and convert a PA to an image rotation angle |
| `PlottingUtilities` | `plot_img` for single images and `wallplot` for a grid of outflow windows |
| `RegionUtilities` | Converts DS9 box regions to outflow parameters (`region_to_params`, `region_to_outflow`), builds source catalogs, and cross-matches guessed driving-source positions against known sources |
| `RegionSelector` | Interactive tool for drawing rotated rectangular regions on an image, used to define outflow windows by hand |

### `compute_dendrogram` parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `noise` | `None` | Image noise. If `None`, it's estimated from the cutout window with sigma clipping |
| `sigma_clip_val` | `3` | Sigma used for clipping when estimating the noise |
| `min_value_factor` | `3.0` | Minimum pixel value to include, in units of the noise |
| `min_delta_factor` | `1.0` | Height above the merge level (in units of the noise) for a leaf to count as its own structure |
| `min_npix` | `100` | Minimum number of pixels for a structure |
| `mask_bright` | `False` | Mask out leaves that contain NaNs or peak above 1000× the noise (e.g. saturated stars), then recompute |

## Example data

`example_images/SerpensMain_NIRCam_f480M.fits` is a JWST NIRCam F480M image of Serpens Main, used in the example notebook
