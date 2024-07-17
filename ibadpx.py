import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
import os
from copy import copy
from matplotlib.widgets import Button, Slider, TextBox, CheckButtons
from matplotlib.ticker import MultipleLocator, MaxNLocator

@mpl.style.context('interactive_badpx_mplstyle.mplstyle')
def fiwdil(Database, concat, badpx_subdir = 'badpx_maps', flag_color='red', zwin=18, clim_init = '0, 99.99%', cmap='gist_yarg', panelsize=(9,9)):
    global badpx_output_dir
    badpx_output_dir = os.path.join(Database.output_dir, badpx_subdir+'/')
    if not os.path.isdir(badpx_output_dir):
        os.makedirs(badpx_output_dir)

    global files
    files = Database.obs[concat]['FITSFILE']

    # Initialize the plot for the first exposure
    global file
    file = files[0]

    global imcube
    imcube = fits.getdata(file)
    
    global immed
    immed = np.nanmedian(imcube, axis=0)

    global f_out
    f_out = badpx_output_dir+file.split('/')[-1].replace('.fits', '_badpixel_map.fits')
    
    global bad_px_map
    try: 
        bad_px_map = fits.getdata(f_out).astype(bool)
    except:
        bad_px_map = np.zeros(imcube.shape, dtype='bool')

    ny,nx = imcube[0].shape

    extent_px = np.array([0,nx,0,ny], dtype=np.float32)-0.5

    cmap = copy(mpl.cm.get_cmap(cmap))
    cmap.set_bad(cmap(0))

    global norm_kwargs
    norm_kwargs=dict(clip=True)
    
    global norm
    norm = mpl.colors.Normalize
    
    global fig, ax, axzoom
    fig, (ax, axzoom) = quick_implot([immed, immed], show=False, norm_kwargs=norm_kwargs, norm=norm, clim=clim_init,
                          lims=np.array([-0.01, 0.01])*nx+extent_px[0:2], ylims=np.array([-0.01, 0.01])*ny+extent_px[2:],
                          cmap=cmap, interpolation='None', panelsize=panelsize, sharex=False, sharey=False)

    ax.set_facecolor(flag_color)
    axzoom.set_facecolor(flag_color)

    ax.set_xlabel('full frame', fontsize=16, labelpad=10)
    axzoom.set_xlabel('zoomed', fontsize=16, labelpad=10)

    children = ax.get_children()
    global implot
    implot = children[np.where([type(child) == mpl.image.AxesImage for child in children])[0][0]]

    zchildren = axzoom.get_children()
    global zimplot
    zimplot = zchildren[np.where([type(child) == mpl.image.AxesImage for child in zchildren])[0][0]]

    x,y = Database.obs[concat]['CRPIX1'][0]-1, Database.obs[concat]['CRPIX2'][0]-1
    axzoom.set(xlim=[x-zwin, x+zwin], ylim=[y-zwin, y+zwin])

    global rect
    rect = mpl.patches.Rectangle((x-zwin, y-zwin), zwin*2, zwin*2, facecolor='None', edgecolor='k', ls='dashed', lw=2)
    ax.add_patch(rect)

    global copied_bad_px_slice
    copied_bad_px_slice = None

    def redraw_bad_px():
        implot.set_alpha(1-bad_px_map[sint.val].astype(float))
        zimplot.set_alpha(1-bad_px_map[sint.val].astype(float))

    def onclick(event):
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        if ((y <= ny-1) and (y >= 0) and (x <= nx-1) and (x >= 0)):
            if event.inaxes == axzoom:
                if check_allints.get_status()[0]: # if all ints is check_allintsed, change all ints based on input
                    bad_px_map[:,y,x] = np.invert(bad_px_map[sint.val, y, x])
                else:
                    bad_px_map[sint.val, y, x] = np.invert(bad_px_map[sint.val, y, x])
                redraw_bad_px()
            elif event.inaxes == ax:
                rect.set_xy((x-zwin, y-zwin))
                axzoom.set(xlim=[x-zwin, x+zwin], ylim=[y-zwin, y+zwin])

    fig.canvas.mpl_connect('button_press_event', onclick)

    fig.subplots_adjust(left=0.1, bottom=0.225, right=0.8, top=0.925)

    global title
    title = fig.text(0.45, 0.95, file.split('/')[-1], ha='center', va='center', fontsize=16)

    ####### Int and exposure sliders #######
    
    def update_int(val):
        implot.set_data(imcube[sint.val])
        zimplot.set_data(imcube[sint.val])
        redraw_bad_px()
        sint.valtext.set_text('')

    def update_exp(val):
        if check_autosave.get_status()[0]:
            save_clicked(None)

        vexp = sexp.val
        sint.set_val(0)

        file = files[vexp]

        global imcube
        imcube = fits.getdata(file)

        global immed
        immed = np.nanmedian(imcube, axis=0)

        global f_out
        f_out = badpx_output_dir+file.split('/')[-1].replace('.fits', '_badpixel_map.fits')
         
        global bad_px_map
        try: 
            bad_px_map = fits.getdata(f_out).astype(bool)
        except:
            bad_px_map = np.zeros(imcube.shape, dtype='bool')

        sint.valstep = np.arange(imcube.shape[0])
        sint.valmax = np.max(sint.valstep)

        ax_int.set_xlim([0, sint.valmax])
        vmin,vmax = str_clim_to_values(imcube, f'{vmin_box.text}, {vmax_box.text}')
        implot.set_norm(norm(vmin, vmax, **norm_kwargs))
        zimplot.set_norm(norm(vmin, vmax, **norm_kwargs))

        if not check_median.get_status()[0]:
            check_median.set_active(0)
        
        im = (immed if check_median.get_status()[0] else imcube[sint.val])
        implot.set_data(im)
        zimplot.set_data(im)

        redraw_bad_px()
        sexp.valtext.set_text('')
        sint.valtext.set_text('')
        title.set_text(file.split('/')[-1])

    global ax_exp
    ax_exp = fig.add_axes([0.1, 0.035, 0.7, 0.05])
    
    global ax_int
    ax_int = fig.add_axes([0.1, 0.135, 0.7, 0.05])

    allowed_exp = np.arange(len(files))
    allowed_int = np.arange(len(imcube))

    global sexp
    sexp = Slider(
        ax_exp, "Exp", allowed_exp.min(), allowed_exp.max(),
        valinit=0, valstep=allowed_exp, color="silver",
        initcolor='None', dragging=True)

    global sint
    sint = Slider(
        ax_int, "Int", allowed_int.min(), allowed_int.max(),
        valinit=0, valstep=allowed_int, color="silver",
        initcolor='None', dragging=True)

    sexp.on_changed(update_exp)
    sint.on_changed(update_int)

    sexp.valtext.set_text('')
    sint.valtext.set_text('')

    ax_exp.add_artist(ax_exp.xaxis)
    ax_int.add_artist(ax_int.xaxis)

    ax_exp.xaxis.set_minor_locator(MultipleLocator(1))
    ax_int.xaxis.set_minor_locator(MultipleLocator(1))
    ax_exp.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_int.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_exp.tick_params(which='both', axis='x', top=False)
    ax_int.tick_params(which='both', axis='x', top=False)

    ax_int.set_visible(False)

    dy = 0.25

    ####### vmin/vmax fields ########
    def vminmax_submit(val):
        vmin, vmax = str_clim_to_values(imcube, f'{vmin_box.text}, {vmax_box.text}')
        implot.set_norm(norm(vmin, vmax, **norm_kwargs))
        zimplot.set_norm(norm(vmin, vmax, **norm_kwargs))

    vmin, vmax = [i.strip() for i in clim_init.split(',')]
    ax_vmin = fig.add_axes([0.825, 0.6+dy, 0.15, 0.04])
    vmin_box = TextBox(ax_vmin, "", initial=vmin, label_pad=0.05)
    ax_vmin.set_title('vmin', fontsize=16)
    vmin_box.on_submit(vminmax_submit)

    ax_vmax = fig.add_axes([0.825, 0.5+dy, 0.15, 0.04])
    vmax_box = TextBox(ax_vmax, "", initial=vmax, label_pad=0.05)
    ax_vmax.set_title('vmax', fontsize=16)
    vmax_box.on_submit(vminmax_submit)

    ####### Save button ########
    def save_clicked(event):
        fits.writeto(f_out, bad_px_map.astype(int), overwrite=True)

    ax_save = fig.add_axes([0.825, 0.4+dy, 0.15, 0.04])
    b_save = Button(ax_save, 'save')
    b_save.on_clicked(save_clicked)

    ####### Copy button ########
    def copy_clicked(event):
        global copied_bad_px_slice
        copied_bad_px_slice = bad_px_map[sint.val].copy()

    ax_copy = fig.add_axes([0.825, 0.3+dy, 0.15, 0.04])
    b_copy = Button(ax_copy, 'copy')
    b_copy.on_clicked(copy_clicked)

    ####### Paste button #######
    def paste_clicked(event):
        if copied_bad_px_slice is not None:
            if check_allints.get_status()[0]:
                bad_px_map[:] = copied_bad_px_slice.copy()
            else:
                bad_px_map[sint.val] = copied_bad_px_slice.copy()
            redraw_bad_px()

    ax_paste = fig.add_axes([0.825, 0.2+dy, 0.15, 0.04])
    b_paste = Button(ax_paste, 'paste')
    b_paste.on_clicked(paste_clicked)

    ####### Clear exposure button #######
    def exp_clear_clicked(event):
        bad_px_map[:,:,:] = 0
        redraw_bad_px()

    ax_eclear = fig.add_axes([0.825, 0.1+dy, 0.15, 0.04])
    b_eclear = Button(ax_eclear, 'clear exp')
    b_eclear.on_clicked(exp_clear_clicked)

    ####### Clear int button #######
    def int_clear_clicked(event):
        if check_allints.get_status()[0]:
            bad_px_map[:,:,:] = 0
        else:
            bad_px_map[sint.val,:,:] = 0
        redraw_bad_px()

    ax_iclear = fig.add_axes([0.825, 0.0+dy, 0.15, 0.04])
    b_iclear = Button(ax_iclear, 'clear int')
    b_iclear.on_clicked(int_clear_clicked)

    # Add toggle to make clicks flag/unflag all integrations for the current exposure
    def allints_clicked(label):
        if not check_allints.get_status()[0]:
            if check_median.get_status()[0]:
                check_median.set_active(0)

    ax_allints = fig.add_axes([0.815, 0.135, 0.15, 0.04])
    [ax_allints.spines[key].set(edgecolor='None') for key in ax_allints.spines]
    ax_allints.set_facecolor('None')
    global check_allints
    check_allints = CheckButtons(
        ax=ax_allints,
        labels=['change all ints'],
        actives=[True],
        label_props={'color': 'k'},
        frame_props={'edgecolor': 'k'},
        check_props={'facecolor': 'k'})

    check_allints.on_clicked(allints_clicked)

    # Add toggle to operate on the median of the ints, always changing the map for all ints at once
    ax_median = fig.add_axes([0.815, 0.085, 0.15, 0.04])
    [ax_median.spines[key].set(edgecolor='None') for key in ax_median.spines]
    ax_median.set_facecolor('None')
    global check_median
    check_median = CheckButtons(
        ax=ax_median,
        labels=['median'],
        actives=[True],
        label_props={'color': 'k'},
        frame_props={'edgecolor': 'k'},
        check_props={'facecolor': 'k'})

    def median_clicked(label):
        if check_median.get_status()[0]:
            ax_int.set_visible(False)
            if not check_allints.get_status()[0]:
                check_allints.set_active(0)
            implot.set_data(immed)
            zimplot.set_data(immed)
        else:
            if check_allints.get_status()[0]:
                check_allints.set_active(0)
            ax_int.set_visible(True)
            implot.set_data(imcube[sint.val])
            zimplot.set_data(imcube[sint.val])

    check_median.on_clicked(median_clicked)

    # Add toggle to autosave when changing exposures
    ax_autosave = fig.add_axes([0.815, 0.035, 0.15, 0.04])
    [ax_autosave.spines[key].set(edgecolor='None') for key in ax_autosave.spines]
    ax_autosave.set_facecolor('None')
    global check_autosave
    check_autosave = CheckButtons(
        ax=ax_autosave,
        labels=['autosave'],
        actives=[True],
        label_props={'color': 'k'},
        frame_props={'edgecolor': 'k'},
        check_props={'facecolor': 'k'}, )

    redraw_bad_px()
    
    def save_buttons(buttons):
        """
        Buttons created in a function call will not work otherwise.
        """
        global button_save
        button_save = buttons
        
    save_buttons([b_save, b_copy, b_paste, b_eclear, b_iclear])


def percentile_clim(im, percentile):
    """
    Compute the color stretch limits for an image based on percentiles.

    Parameters:
    ----------
    im : array-like
        The input image.
    percentile : float or list of float
        The percentile(s) to use for computing the color stretch limits. If a single value is provided, a symmetric
        color stretch is generated spanning plus and minus the P-percentile of the absolute value of im. If two values
        are provided, they are used as the lower and upper limit percentiles.

    Returns:
    -------
    clim : array
        The lower and upper limits of the color stretch.
    """
    vals = np.unique(im)
    if np.isscalar(percentile) or len(percentile) == 1:
        clim = np.array([-1,1])*np.nanpercentile(np.abs(vals), percentile)
    else:
        clim = np.nanpercentile(vals, percentile)
    return clim


def quick_implot(im, clim=None, clim_perc=[1.0, 99.0], cmap=None,
                 show_ticks=False, lims=None, ylims=None,
                 norm=mpl.colors.Normalize, norm_kwargs={},
                 figsize=None, panelsize=[5,5], fig_and_ax=None, extent=None,
                 show=True, tight_layout=True, alpha=1.0,
                 cbar=False, cbar_orientation='vertical',
                 cbar_kwargs={}, cbar_label=None, cbar_label_kwargs={},
                 interpolation=None, sharex=True, sharey=True,
                 save_name=None, save_kwargs={}):
    """
    Plot an image or set of images with customizable options.

    Parameters:
    ----------
    im : array-like
        The input image(s) to plot. If im is a 2D array, a single panel will be created. If im is a 3D array, a row of
        panels will be created. If im is a 4D array, a grid of panels will be created. E.g., 
        im=[[im1, im2], [im3, im4], [im5, im6]] will create a plot with 3 rows and 2 columns.
    clim : str or tuple, optional
        The color stretch limits. If a string is provided, it should contain a comma-separated pair of values.
        If a tuple is provided, it should contain the lower and upper limits of the color stretch.
    clim_perc : float or list of float, optional
        The percentile(s) to use for computing the color stretch limits. If a single value is provided, a symmetric
        color stretch is generated spanning plus and minus the P-percentile of the absolute value of im. If two values
        are provided, they are used as the lower and upper limit percentiles.
    cmap : str or colormap, optional
        The colormap to use for the image.
    show_ticks : bool, optional
        Whether to show ticks on the plot.
    lims : tuple, optional
        The x-axis (and y-axis if ylims is not provided) limits of the plot.
    ylims : tuple, optional
        The y-axis limits of the plot.
    norm : matplotlib.colors.Normalize or subclass, optional
        The normalization class to use for the color mapping.
    norm_kwargs : dict, optional
        Additional keyword arguments to pass to the normalization class.
    figsize : tuple, optional
        The size of the figure in inches. If not provided, the size will be determined based on the number of panels and
        the panelsize argument.
    panelsize : list, optional
        The size of each panel in the figure. 
    fig_and_ax : tuple, optional
        A tuple containing a matplotlib Figure and Axes object to use for the plot.
    extent : array-like, optional
        The extent of the plot as [xmin, xmax, ymin, ymax].
    show : bool, optional
        Whether to show the plot or return the relevant matplotlib objects.
    tight_layout : bool, optional
        Whether to use tight layout for the plot.
    alpha : float, optional
        The transparency of the image.
    cbar : bool, optional
        Whether to show a colorbar.
    cbar_orientation : str, optional
        The orientation of the colorbar ('vertical' or 'horizontal').
    cbar_kwargs : dict, optional
        Additional keyword arguments to pass to the colorbar.
    cbar_label : str, optional
        The label for the colorbar.
    cbar_label_kwargs : dict, optional
        Additional keyword arguments to pass to the colorbar label.
    interpolation : str, optional
        The interpolation method to use with imshow.
    sharex : bool, optional
        Whether to share the x-axis among subplots.
    sharey : bool, optional
        Whether to share the y-axis among subplots.
    save_name : str, optional
        The filename to save the plot. Plot will be saved only if this argument is provided.
    save_kwargs : dict, optional
        Additional keyword arguments to pass to the save function.

    Returns:
    -------
    fig : matplotlib Figure
        The created Figure object.
    ax : matplotlib Axes or array of Axes
        The created Axes object(s).
    cbar : matplotlib Colorbar, optional
        The created Colorbar object.

    Notes:
    ------
    - If clim is a string, it should contain a comma-separated pair of values. The values can be interpretable as floats,
      in which case they serve as the corresponding entry in the utilized clim. Alternatively, they can contain a '%'
      symbol, in which case they are used as a percentile bound. For example, clim='0, 99.9%' will yield an image with a
      color stretch spanning [0, np.nanpercentile(im, 99.9)]. If clim contains a '*', the options can be multiplied.
    - If clim is not provided, the color stretch limits will be computed based on the clim_perc parameter.
    - If clim_perc is a single value, the lower and upper limits of the color stretch will be symmetric.
    - If clim_perc is a list of two values, they will be used as the lower and upper limit percentiles.
    """
    if isinstance(clim, str):
        s_clim = [i.strip() for i in clim.split(',')]
        clim = []
        for s in s_clim:
            if s.isdigit():
                clim.append(float(s))
            elif '%' in s:
                if '*' in s:
                    svals = []
                    for si in s.split('*'):
                        if '%' in si:
                            svals.append(np.nanpercentile(im, float(si.replace('%',''))))
                        else:
                            svals.append(float(si))
                    clim.append(np.prod(svals))
                else:
                    clim.append(np.nanpercentile(im, float(s.replace('%',''))))
            else:
                raise ValueError(
                    """
                    If clim is a string, it should contain a comma separating
                    two entries. These entries should be one of:
                    a) interpretable as a float, in which case they serve as the 
                    corresponding entry in the utilized clim, b) they should contain a
                    % symbol, in which case they are used as a percentile bound;
                    e.g., clim='0, 99.9%' will yield an image with a color
                    stretch spanning [0, np.nanpercentile(im, 99.9)], or c) they
                    should contain a '*' symbol, separating either of the 
                    aforementioned options, in which case they will be multiplied.
                    """)
            
    elif clim is None:
        clim = percentile_clim(im, clim_perc)
        
    if ylims is None:
        ylims = lims
        
    normalization = norm(vmin=clim[0], vmax=clim[1], **norm_kwargs)

    if np.ndim(im) in [2,3,4]:
        im_4d = np.expand_dims(im, np.arange(4-np.ndim(im)).tolist()) # Expand dimensions to 4D if not already to easily extract nrows and ncols
        nrows, ncols = im_4d.shape[0:2]
    else:
        raise ValueError("Argument 'im' must be a 2, 3, or 4 dimensional array")
    n_ims = nrows * ncols

    if fig_and_ax is None:
        if figsize is None:
            figsize = np.array([ncols,nrows])*np.asarray(panelsize)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        fig, ax = fig_and_ax

    if n_ims == 1:
        ax, im = [ax], [im.squeeze()]
    else:
        im = np.asarray(im).reshape((nrows*ncols, *np.shape(im)[-2:]))
        ax = np.asarray(ax).flatten()

    for ax_i, im_i in zip(ax, im):
        implot = ax_i.imshow(im_i, origin='lower', cmap=cmap, norm=normalization, extent=extent, alpha=alpha, interpolation=interpolation)
        if not show_ticks:
            ax_i.set(xticks=[], yticks=[])
        ax_i.set(xlim=lims, ylim=ylims)
    if tight_layout:
        fig.tight_layout()
    if cbar:
        cbar = fig.colorbar(implot, ax=ax, orientation=cbar_orientation, **cbar_kwargs)
        cbar.set_label(cbar_label, **cbar_label_kwargs)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', **save_kwargs)
    if show:
        plt.show()
        return None
    if n_ims == 1:
        ax = ax[0]
    if cbar:
        return fig, ax, cbar
    return fig, ax


def str_clim_to_values(im, clim):
    s_clim = [i.strip() for i in clim.split(',')]
    clim = []
    for s in s_clim:
        try:
            clim.append(float(s))
        except:
            if '*' in s:
                svals = []
                for si in s.split('*'):
                    if '%' in si:
                        svals.append(np.nanpercentile(im, float(si.replace('%',''))))
                    else:
                        svals.append(float(si))
                clim.append(np.prod(svals))
            else:
                clim.append(np.nanpercentile(im, float(s.replace('%',''))))
    return clim