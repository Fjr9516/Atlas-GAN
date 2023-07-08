#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
everything related to visualization of 2D/3D images is here
'''
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk

# itk image
def rescale(itkimg):
    minmaxfilter = sitk.MinimumMaximumImageFilter()
    minmaxfilter.Execute(itkimg)
    max_value = minmaxfilter.GetMaximum()
    min_value = minmaxfilter.GetMinimum()

    return sitk.IntensityWindowing(itkimg, windowMinimum=min_value, windowMaximum=max_value,
                                            outputMinimum=0.0, outputMaximum=1)

# numpy image
def crop_ndarray(ndarr_img,  uppoint=[0, 13, 13], out_size = [160, 160, 192], show = False):
    dim1 = uppoint[0]
    dim2 = uppoint[1]
    dim3 = uppoint[2]
    template = ndarr_img[dim1:(dim1 + out_size[0]), dim2:(dim2 + out_size[1]), dim3:(dim3 + out_size[2])]
    # plot
    mid_slices_moving = [np.take(template, template.shape[d] // 2, axis=d) for d in range(3)]
    mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
    mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)
    slices(mid_slices_moving, cmaps=['gray'], grid=[1, 3], save=False, show=show)

def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           suptitle = None,
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           axes_off=True,
           save=False, # option to save plot
           save_path=None, # save path
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    taken from voxelmorph + small modification
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    if show:
        plt.tight_layout()
        plt.show()

    if save:
        plt.savefig(save_path + '.png')
    return (fig, axs)

def load_nii(path, way = 'nibabel'):
    '''
    Args:
        path: path of nifti file
        way: nibabel or sitk, sitk follow the same orientation of numpy load a npz
    Returns: imf
    '''
    if way == 'nibabel':
        img = nib.load(path).get_fdata()
    elif way == 'sitk':
        load_img = sitk.ReadImage(path, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(load_img)
    else:
        raise ValueError('way expected to be nibabel or sitk')
    return(img)

def correct_vox2ras_matrix(wrong_nifit_path, save_nifit_path = None, reference_nifiti = './src/align_norm.nii.gz'):
    '''

    Args:
        wrong_nifit_path: wrong nifiti file, mainly caused by using simpleitk, it uses a
                          different loading principle, need to use it to convert back.
                          reference web page: https://itk.org/pipermail/community/2017-November/013783.html
        save_nifit_path: Can override the old one, or save as another name and/or path
        reference_nifiti: an OASIS3 data which has right orientation

    Returns:
        saved_corrected_nifit_file
    '''
    real_img = sitk.ReadImage(reference_nifiti)
    if wrong_nifit_path.endswith('.nii.gz') or wrong_nifit_path.endswith('.nii'):
        wrong_img_npy = nib.load(wrong_nifit_path).get_fdata()
    else:
        raise ValueError('Input wrong_nifit_path is not a string!')

    wrong_img = sitk.GetImageFromArray(wrong_img_npy)
    wrong_img.SetSpacing(real_img.GetSpacing())
    origin = []
    for x, y in zip(wrong_img.GetSize(), real_img.GetOrigin()):
        if y > 0:
            origin.append(1 * x / 2.0)
        else:
            origin.append(-1 * x / 2.0)
    wrong_img.SetOrigin(origin)
    wrong_img.SetDirection(real_img.GetDirection())

    if save_nifit_path == None:
        sitk.WriteImage(wrong_img, wrong_nifit_path)
    else:
        sitk.WriteImage(wrong_img, save_nifit_path)