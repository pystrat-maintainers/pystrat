# import standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

# python things
from importlib import resources
import warnings

# plotting
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from PIL import Image

##
## Global vars
##
mod_dir = os.path.dirname(os.path.realpath(__file__))


###############
### CLASSES ###
###############


class Fence:
    """Fence diagram class.

    Organizes sections according to a shared datum.

    This class permits plotting of fence diagrams.

    Parameters
    ----------
    sections : 1d array_like
        List of Sections to be put into the fence
    
    datums : 1d array_like, optional
        If not specified, the datum for each section will be the bottom. If specified, 
        must be list of same length as number of sections with heights in each section for the datum.

    correlations : 2d array_like, optional
        Each column is a correlated horizon where the rows are the heights of this horizon in each section.
        will plot as a line between fence posts. Default is no correlations.

    coordinates : 1d array_like, optional
        1D coordinates of sections, reflecting distance between them. Distances between sections will be used to scale the plotting distances 
        in the fence diagram.

    Attributes
    ----------
    n_sections : int
        Number of sections in the fence diagram

    sections : 1d array_like
        List of sections in the fence diagram
    
    datums : 1d array_like
        Datum for each section

    correlations : 2d array_like
        Correlated horizons

    coordinates : 1d array_like
        Coordinates of sections
    """

    def __init__(self,
                 sections,
                 datums=None,
                 correlations=None,
                 coordinates=None):
        """Initialize fence.
        """
        self.n_sections = len(sections)
        self.sections = [copy.deepcopy(section) for section in sections]
        
        # if no datums provided, then assume bottom of each section is datum
        if datums is None:
            datums = np.zeros(self.n_sections)
            for ii in range(self.n_sections):
                datums[ii] = sections[ii].base_height[0]
        else:
            assert len(datums) == self.n_sections, \
                'Number of datums should equal number of sections'
        self.datums = datums

        if correlations is not None:
            assert correlations.shape[0] == self.n_sections, \
                'Number of correlated horizons should match number of sections'
        self.correlations = correlations

        # if coordinates not provided, then assume sections are equally spaced
        if coordinates is None:
            self.coordinates = np.cumsum(np.ones(self.n_sections))
        else:
            assert len(coordinates) == self.n_sections, \
                'Number of section distances should match number of sections'
            self.coordinates = coordinates

        # order sections, correlations, datums by coordinates
        idx = np.argsort(self.coordinates)
        self.coordinates = self.coordinates[idx]
        self.sections = [self.sections[x] for x in idx]
        self.datums = [self.datums[x] for x in idx]
        if self.correlations is not None:
            self.correlations = self.correlations[idx]

        # apply datums as shift to sections and correlations
        for ii in range(self.n_sections):
            self.sections[ii].shift_heights(-self.datums[ii])
            if self.correlations is not None:
                self.correlations[ii, :] = self.correlations[ii, :] - \
                                            self.datums[ii]

    def plot(self,
             style,
             fig=None,
             sec_wid_fac=1,
             col_buffer_fac=0.2,
             distance_spacing=False,
             plot_distances=None,
             distance_labels=False,
             distance_labels_style=None,
             plot_correlations=None,
             data_attributes=None,
             data_attribute_styles=None,
             section_plot_style=None,
             sec_names_rotate=True,
             sec_names_fontsize=10,
             **kwargs):
        """Plot a fence diagram

        Parameters
        ----------
        style : Style
            A Style object.

        fig : matplotlib.figure.Figure, optional
            Figure to plot into if desired, by default None. If None, will create and return a new figure.

        sec_wid_fac : float, optional
            Ratio of section axis width to data attribute axes widths, defaults to 1. A value of 1 means that the section axis is the same width as the data attribute axes. Values less than 1 mean that the section axis is narrower than the data attribute axes.

        col_buffer_fac : float, optional
            Fraction of section width used to buffer between columns in the fence, defaults to 0.2. A value of 0 means no buffer and columns immediately abut each other.

        distance_spacing : boolean, optional
            Whether or not to scale the distances between sections according to the distances between self.coordinates, or plot_distances, if set. Default is False. If False, then sections are equally spaced.

        plot_distances : 1d array-like, optional
            Distances between sections to use for plotting. Default is None. If None, then distances are calculated from coordinates. If set, then length (n_sections - 1).

        distance_labels : 1d array-like or boolean, optional
            Labeling of distances between sections. Default is False. If False, no labels are plotted. If True, labels are plotted with the actual distances between sections (based on coordinates). If an array-like, then length must be (n_sections - 1) and values specify manual labeling of distances.

        distance_labels_style : dictionary, optional
            Style dictionary for distance labels. Default is None. If None, a default style is used. Dictionary is passed to matplotlib.pyplot.annotate.

        plot_correlations : boolean, optional
            Whether or not to plot correlated horizons. Default is True; this parameter is ignored if correlations is None.

        data_attributes : 1d array-like, optional
            List of data attributes to plot. Default is None. If None, no data attributes are plotted. If the attribute is not defined for a particular section, it is not plotted.

        data_attribute_styles : 1d array-like, optional
            Style dictionary or dictionaries to use to plot data attributes. Defaults to None, in which case a default style is used. Either same length as data_attributes, or length of one. If length of one, then the same style is used for all data attributes.
        
        section_plot_styles : dictionary, optional
            Dictionary of style parameters passed to section plotting. Default is None. If None, a default style is used.

        sec_names_rotate : boolean, optional
            whether to plot section names vertically (True) or horizontally above columns in fence. Default is True.

        sec_names_fontsize : float, optional
            Fontsize for section names. Default is 10.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            Returned if no figure is provided

        axes : list
            List of matplotlib axes objects for each column.

        axes_dat : list
            List of matplotlib axes objects for data attribute. Returned if data_attributes is not None. Each entry in the list is a list of axes for each data attribute. Sections with no data attributes will have a None entry.
        """
        # before setting anything up, need to know if we're plotting data attributes and how many
        # number of attributes to plot per section
        n_att_sec = np.zeros(self.n_sections).astype(int)
        if data_attributes is not None:
            for ii in range(self.n_sections):
                for attribute in data_attributes:
                    if hasattr(self.sections[ii], attribute):
                        n_att_sec[ii] = n_att_sec[ii] + 1
            assert np.sum(n_att_sec) > 0, 'data attribute not found in any sections'
            # update sec_width to reflect the number of data attributes being plotted
            # sec_wid = sec_wid/np.max(n_att_sec)

            # set up a default style if none supplied
            if data_attribute_styles is None:
                data_attribute_styles = np.max(n_att_sec) * \
                                        [{'marker': '.', 
                                         'color': 'k',
                                         'linestyle': ''}]
            elif len(data_attribute_styles) == 1:
                data_attribute_styles = np.max(n_att_sec) * data_attribute_styles

        # set up axes to plot sections into (will share vertical coordinates)
        if fig == None:
            fig_provided = False
            fig = plt.figure(**kwargs)
        else:
            fig_provided = True
        min_height = np.min(
            [section.base_height[0] for section in self.sections])
        max_height = np.max(
            [section.top_height[-1] for section in self.sections])

        '''
        The Fence class conceptualizes section geometry as in the following diagram.
        ┌──────┬──┬──┐b┌──────┬──┐b┌──────┐
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │      │  │  │ │      │  │ │      │
        │  x   │a │a │ │  x   │a │ │  x   │
        └──────┴──┴──┘ └──────┴──┘ └──────┘
                                            
        └────────────┘ └─────────┘ └──────┘
            cw1          cw2        cw3    

        x: width of each section axis (same for all sections)
        a: width of each data attribute axis (same for all data attributes)
        b: buffer space between section axes. the minimum buffer is set by tau*x
            tau: fraction of section width that is buffer space, tau is col_buffer_fac                    
        cw1: width of column 1 (similarly for cw2, cw3)

        The sum over cw's and b's is 1. 
        x is related to a as follows:
            x/a = gamma where gamma is sec_wid_fac
        When plot distances (dist) vary, the smallest plot distance is set to tau*x, and all other plot distances are scaled accordingly.
            bi = dist/dist_min * tau*x

        Yields the following equation in x:
        n*x + m/gamma*x + sum(di)*tau*x = 1
        where n is the number of sections, m is the total number of data attributes present across all sections, and di is the scaled distance between sections i and i+1 as defined above.

        x is then:
        x = 1/(n + m/gamma + sum(di)*tau)
        '''

        # set up buffer distance between columns
        if distance_spacing:
            # distances between sections
            if plot_distances is None:
                distances = np.diff(self.coordinates)
            else:
                assert len(plot_distances) == (
                    self.n_sections -
                    1), 'length of plot_distances must be n_sections - 1'
                distances = plot_distances
        
        # uniform buffer spacing if not specified
        else:
            distances = np.ones(self.n_sections - 1)

        # solve for x, a, and bi
        dist_norm = distances / np.min(distances) # di above
        m = np.sum(n_att_sec) # m above
        sec_wid = 1 / (self.n_sections + m/sec_wid_fac + np.sum(dist_norm)*col_buffer_fac)  # x above
        data_att_wid = sec_wid / sec_wid_fac # a above
        bi = dist_norm * col_buffer_fac * sec_wid # bi above
        col_widths = sec_wid + n_att_sec * data_att_wid # cw above

        # compute left coordinates of section axes
        ax_left_coords = np.cumsum(np.insert(col_widths, 0, 0))[0:-1] + \
                         np.cumsum(np.insert(bi, 0, 0))

        # set up section axes
        axes = []
        for ii in range(self.n_sections):
            axes.append(plt.axes([ax_left_coords[ii], 0, sec_wid, 1],))

        # set up data attribute axes
        if data_attributes is not None:
            axes_dat = []
            for ii in range(self.n_sections):
                # dont make an axis if the section doesn't have any data attributes
                if n_att_sec[ii] == 0:
                    axes_dat.append(None)
                    continue
                else:
                    cur_sec_dat_axes = []
                for jj in range(len(data_attributes)):
                    if hasattr(self.sections[ii], data_attributes[jj]):
                        cur_sec_dat_axes.append(
                            plt.axes([ax_left_coords[ii] + sec_wid + jj * data_att_wid, 0,
                                        data_att_wid, 1]))
                    else:
                        cur_sec_dat_axes.append(None)
                axes_dat.append(cur_sec_dat_axes)

        # enforce axis limits before plotting to maintain swatch scaling
        for ii in range(self.n_sections):
            axes[ii].set_xlim([0, 1])
            axes[ii].set_ylim([min_height, max_height])

        # also enforce axis limits for data attributes
        if data_attributes is not None:
            for ii in range(self.n_sections):
                for jj, attribute in enumerate(data_attributes):
                    if hasattr(self.sections[ii], attribute):
                        axes_dat[ii][jj].set_ylim([min_height, max_height])

        # then plot sections
        if section_plot_style is None:
            section_plot_style = {}
        for ii, section in enumerate(self.sections):
            section.plot(style, ax=axes[ii], **section_plot_style)

        # the plot data attributes, if specified
        if data_attributes is not None:
            for ii in range(self.n_sections):
                for jj, attribute in enumerate(data_attributes):
                    if hasattr(self.sections[ii], attribute):
                        self.sections[ii].plot_data_attribute(attribute, 
                                                              ax=axes_dat[ii][jj],
                                                              style=data_attribute_styles[jj])

        # then move and scale axes so that vertical coordinate is consistent and datum is at same height in figure
        for ii in range(self.n_sections):
            if sec_names_rotate:
                axes[ii].set_title(self.sections[ii].name, rotation=90, ha='right', fontsize=sec_names_fontsize)
            else:
                axes[ii].set_title(self.sections[ii].name, fontsize=sec_names_fontsize)

        # clean up plotting sections
        for ii in range(1, self.n_sections):
            # axes[ii].set_yticklabels = []
            # axes[ii].set_ylabel = None
            axes[ii].get_yaxis().set_visible(False)
            axes[ii].spines['left'].set_visible(False)
        for ii in range(self.n_sections):
            axes[ii].get_xaxis().set_visible(False)
            axes[ii].spines['bottom'].set_visible(False)

        # plot correlated beds as connections
        if self.correlations is not None and plot_correlations is None:
            plot_correlations = True
        else:
            plot_correlations = False
        if plot_correlations:
            for ii in range(self.correlations.shape[1]):
                for jj in range(self.n_sections - 1):
                    # need to account for data attribute axes
                    if data_attributes is not None:
                        # point needs to account for the width of the column up to the rightmost data attribute axis
                        if n_att_sec[jj] > 0:
                            # need x-coord of max x-value in rightmost data attribute axes
                            # x-coord has to be transformed to jth section axes coordinate system
                            x_right_dat = axes_dat[jj][-1].get_xlim()[1] # coordinate in data attribute axes
                            x_right_disp = axes_dat[jj][-1].transData.transform((x_right_dat, 0)) # coordinate in display axes
                            x_right = axes[jj].transData.inverted().transform(x_right_disp)[0] # coordinate in section axes
                            xyA = [x_right, self.correlations[jj, ii]]

                        else:
                            xyA = [axes[jj].get_xlim()[1],
                                self.correlations[jj, ii]]
                        # don't forget about unit labels
                        xyB = [axes[jj].get_xlim()[0], self.correlations[jj + 1, ii]]
                    else:
                        xyA = [axes[jj].get_xlim()[1], self.correlations[jj, ii]]
                        # xyB = [0, self.correlations[jj + 1, ii]]
                        xyB = [axes[jj].get_xlim()[0], self.correlations[jj + 1, ii]]
                    if np.any(np.isnan(xyA)) or np.any(np.isnan(xyB)):
                        continue
                    con = ConnectionPatch(xyA=xyA,
                                          coordsA=axes[jj].transData,
                                          xyB=xyB,
                                          coordsB=axes[jj + 1].transData, zorder=15)
                    fig.add_artist(con)

        # plot datum
        for jj in range(self.n_sections - 1):
            xyA = [1, 0]
            xyB = [0, 0]
            con = ConnectionPatch(xyA=xyA,
                                  coordsA=axes[jj].transData,
                                  xyB=xyB,
                                  coordsB=axes[jj + 1].transData)
            fig.add_artist(con)

        # label distances between sections
        if distance_labels == False:
            pass
        elif (distance_labels == True) or (len(distance_labels) > 0):
            # figure out styling
            distance_labels_style_default = {'fontsize': 9,
                                             'ha': 'center'}
            if distance_labels_style is None:
                distance_labels_style = distance_labels_style_default
            else:
                distance_labels_style = {**distance_labels_style_default,
                                         **distance_labels_style}

            # validate distance_labels
            if type(distance_labels) == bool:
                distance_labels = np.diff(self.coordinates)
            else:
                assert len(distance_labels) == (
                    self.n_sections - 1), 'incorrect number of distance labels'

            fig_bbox = fig.get_tightbbox(
                fig.canvas.get_renderer())._bbox.get_points()
            x_offset = fig_bbox[0, 0]
            cur_y_coord = (fig_bbox[1, 1] - fig_bbox[0, 1])
            wid = fig_bbox[1,0]
            # hei = fig_bbox[1,1]
            for ii, distance in enumerate(distance_labels):

                cur_ax_bbox = axes[ii].get_window_extent().get_points()
                nex_ax_bbox = axes[ii + 1].get_window_extent().get_points()
                cur_x_coord = nex_ax_bbox[0, 0] - (
                    nex_ax_bbox[0, 0] - cur_ax_bbox[1, 0]) / 2 - x_offset

                # this value is the factor to account for the fact that the
                # figure coordinates need to be larger than one for the y
                # coordinate by the amount that the figur is larger than the 
                # axis bounding box
                vert_axis_fact = (cur_y_coord-cur_ax_bbox[1,1])/(2*cur_ax_bbox[1,1]) + 1
                plt.annotate(distance, (cur_x_coord/wid, vert_axis_fact),
                             xycoords='figure fraction',
                             **distance_labels_style)

        # figure out what to return
        if not fig_provided:
            if data_attributes is not None:
                return fig, axes, axes_dat
            return fig, axes
        if data_attributes is not None:
            return axes, axes_dat
        return axes


class Section:
    """Stratigraphic section class.

    Organizes all data associated with a single stratigraphic section.
    
    Parameters
    ----------
    thicknesses : 1d array_like
        Thicknesses of the facies. Any NaNs will be automatically removed.

    facies : 1d array_like
        Observed facies. Any NaNs will be automatically removed.

    name : str, optional
        Name of the section. Default is None.

    annotations : pandas.DataFrame, optional
        DataFrame with columns 'height' and 'annotation'. Heights correspond to stratigraphic heights at which to plot annotations. Annotations are .png files in the annotations folder. Default is None.

    units : numpy.ndarray, optional
        Array of unit names for labeling intervals such as groups, formations, members, etc. Each bed in the section must have a name. Each column in units corresponds to a level of labeling (e.g. group, formation, member). Long dimension of the array must be len(thicknesses). Default is None.
    
    Attributes
    ----------
    top_height : 1d array_like
        Top height of each unit.

    base_height : 1d array_like
        Base height of each unit.

    unit_number : 1d array_like
        Number of each unit.
    
    n_beds : int
        Number of beds in the section.

    total_thickness : float
        Total thickness of the section.

    unique_facies : 1d array_like
        Unique facies in the section.

    n_unique_facies : int
        Number of unique facies in the section.
    """

    def __init__(self, thicknesses, facies, 
                 name=None, annotations=None, units=None):
        """Initialize Section
        """
        # convert to arrays and check the dimensionality
        thicknesses = attribute_convert_and_check(thicknesses)
        facies = attribute_convert_and_check(facies)
        if annotations is not None:
            assert 'height' in annotations.columns, \
                  'annotations must have a height column'
            assert 'annotation' in annotations.columns, \
                'annotations must have an annotation column'
            # if DataFrame is empty, set to None
            if annotations.empty:
                annotations = None

        # check that the thicknesses are numeric
        # if thicknesses.dtype == np.object:
        #     raise Exception('Thickness data must be floats or ints.')

        # check for NaNs, and get rid of them
        thicknesses_nan_mask = np.isnan(thicknesses)
        if np.sum(thicknesses_nan_mask) > 0:
            warnings.warn(
                'Thickness data contains NaNs. These rows will be '
                'automatically removed, but you should check to make '
                'sure that this is appropriate for your dataset.')
            thicknesses = thicknesses[~thicknesses_nan_mask]

        facies_nan_mask = pd.isnull(facies)
        if np.sum(facies_nan_mask) > 0:
            warnings.warn(
                'Facies data contains NaNs. These rows will be '
                'automatically removed, but you should check to make '
                'sure that this is appropriate for your dataset.')
            facies = facies[~facies_nan_mask]

        # check that the length of the thicknesses and facies match
        n_thicknesses_units = len(thicknesses)
        n_facies_units = len(facies)
        if n_thicknesses_units != n_facies_units:
            raise Exception('Length of thicknesses and facies data are not '
                            'equal. If a warning was raised regarding the '
                            'presence of NaNs in your data, their presence '
                            'may be the cause of this mismatch.')
        
        # save units
        if units is None:
            self.units = None
        else:
            units = np.atleast_2d(units)
            # make longest dimension into rows
            if units.shape[0] < units.shape[1]:
                units = units.T
            assert units.shape[0] == n_thicknesses_units, \
                'Unit names need to match number of thicknesses.'
            self.units = units

        # assign the core data to attributes
        self.thicknesses = thicknesses
        self.facies = facies
        self.annotations = annotations

        # add some other facies attributes
        self.top_height = np.cumsum(thicknesses)
        self.base_height = np.cumsum(thicknesses) - thicknesses
        self.unit_number = np.arange(n_thicknesses_units)

        # add some generic attributes
        self.name = name
        self.n_beds = n_thicknesses_units
        self.total_thickness = np.sum(thicknesses)
        self.unique_facies = np.unique(facies)
        self.n_unique_facies = len(np.unique(facies))

        # keep track of attributes
        self.facies_attributes = [
            'unit_number', 'thicknesses', 'base_height', 'top_height', 'facies'
        ]
        self.generic_attributes = [
            'name', 'n_beds', 'total_thickness', 'unique_facies',
            'n_unique_facies'
        ]
        self.data_attributes = []

    def shift_heights(self, shift):
        """
        Shift all heights by some fixed distance. (additive)

        Parameters
        ----------
        shift : int or float
            Amount by which to shift stratigraphic heights
        """
        # update facies attributes
        self.base_height = self.base_height + shift
        self.top_height = self.top_height + shift

        # data attributes
        for attribute in self.data_attributes:
            data_attribute = getattr(self, attribute)
            data_attribute.height = data_attribute.height + shift
            setattr(self, attribute, data_attribute)
        
        # annotation heights
        if self.annotations is not None:
            self.annotations.height = self.annotations.height + shift

    def plot_data_attribute(self, attribute, ax=None, style=None, clean=True):
        """Plot a data attribute

        Parameters
        ----------
        attribute : str
            Name of the data attribute to plot
        ax : matplotlib.axes._axes.Axes, optional
            Axis to plot into, by default None. If None, will create a new axis and return it.
        style : dict, optional
            Plotting style dictionary compatible with matplotlib.pyplot.plot, by default None. If None, a default style will be used.
        clean : bool, optional
            Whether or not to clean up the axes, by default True. If True, will remove the y-axis and the top and right spines.

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            Returned if no axis is provided
        """
        if ax is None:
            ax = plt.axes()

        default_style = {'marker': '.',
                         'markersize': 5,
                         'color': 'k',
                         'linestyle': ''}
        if style is None:
            style = default_style
        else:
            style = {**default_style, **style}
        assert hasattr(self, attribute), 'Section does not have requested attribute.'

        cur_att = getattr(self, attribute)
        ax.plot(cur_att.values, cur_att.height, **style)
        ax.set_xlabel(attribute)
        # clean up axes
        if clean:
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if ax is not None:
            return ax


    def plot(self, style, 
             ax=None, 
             linewidth=1, 
             annotation_height=0.15,
             label_units=False,
             unit_label_wid_tot=0.2,
             unit_fontsize=8):
        """
        Plot this section using a Style object.

        Parameters
        ----------
        style : Style
            A Style object.

        ax : matplotlib axis
            Axis in which to plot, if desired. Otherwise makes a new axes object.
            AXIS LIMITS MUST BE APPLIED IN ADVANCE FOR SWATCHES TO PLOT CORRECTLY

        linewidth : float
            The linewidth when drawing the stratigraphic section.

        annotation_height : float, optional
            The height in inches for annotation graphics. Defaults to 0.15. Set to zero to not plot annotations.

        label_units : boolean, optional
            Whether or not to label units on the left. Default is False. If True, then section must have unit names specified.

        unit_label_wid_tot : float, optional
            Fractional width of the space to label units (if provided) on the left of the column. Default is 0.2.

        unit_fontsize : float, optional
            Fontsize for labeling units. Default is 8.

        """
        # get the attributes - implicitly checks if the attributes exist
        if not self.style_compatibility(style):
            raise ValueError('Style is not compatible with section.')

        # initialize
        if ax == None:
            ax = plt.axes()
            ax.set_ylim([self.base_height[0], self.top_height[-1]])

        # set up x axis
        if label_units:
            ax.set_xlim([-unit_label_wid_tot, 1])

        # determine the axis height and limits first
        # ax_height = style.height_scaling_factor * section.total_thickness

        # initiate counting of the stratigraphic height
        strat_height = self.base_height[0]

        # loop over elements of the data
        for i in range(self.n_beds):

            # pull out the thickness
            this_thickness = self.thicknesses[i]

            # loop over the elements in Style to get the color and width
            for j in range(style.n_labels):
                if self.facies[i] == style.labels[j]:
                    this_color = style.color_values[j]
                    this_width = style.width_values[j]

            # create the rectangle
            ax.add_patch(
                Rectangle((0.0, strat_height),
                          this_width,
                          this_thickness,
                          facecolor=this_color,
                          edgecolor='k',
                          linewidth=linewidth))

            # if swatch is defined, plot it
            # if style.swatch_values[0] != None:
            ax.autoscale(False)
            if style.swatch_values is not None:
                for j in range(style.n_labels):
                    if self.facies[i] == style.labels[j]:
                        this_swatch = style.swatch_values[j]
                        if this_swatch == 0:
                            continue
                        extent = [
                            0, this_width, strat_height,
                            strat_height + this_thickness
                        ]
                        plot_swatch(this_swatch, extent, ax,
                                    swatch_wid=style.swatch_wid)             

            # count the stratigraphic height
            strat_height = strat_height + this_thickness

        # plot annotations
        if (self.annotations is not None) and (annotation_height != 0) \
            and (style.annotations is not None):
            # keep only annotations with symbols in style
            annotations = []
            heights = []
            for ii, annotation in enumerate(self.annotations.annotation):
                if annotation not in list(style.annotations.keys()):
                    warnings.warn(f'Annotation {annotation} not in style.')
                else:
                    annotations.append(annotation)
                    heights.append(self.annotations.height.values[ii])

            # preprocess annotation heights to determine overlap of symbols
            
            # get height in data coordinates
            height = annotation_height/get_inch_per_dat(ax)[1]
            # these are the bottom coordinates for each annotation
            bot_coords = heights - height/2
            # these are the top coordinates
            top_coords = bot_coords + height

            # warn user if annotations will overlap (i.e. any top coordinate is between another bottom and top coordinate)
            for ii, top_coord in enumerate(top_coords):
                if np.any((top_coord > bot_coords) & (top_coord < top_coords)):
                    warnings.warn(f'Annotation at height {self.annotations.height.values[ii]} will overlap with another annotation. Consider decreasing annotation_height.')

            # iterate over the annotations
            for ii, annotation in enumerate(annotations):
                # remember to adjust starting position for the number of annotations
                pos = [1.1, bot_coords[ii]]
                # check that annotation is in style
                plot_annotation(style.annotations[annotation], pos, height, ax)

        # label units
        if label_units and (self.units is not None):
            # number of unit levels
            n_levels = self.units.shape[1]
            # width of each hierarchical label set
            unit_label_wid = unit_label_wid_tot / n_levels

            # loop over each level
            for ii in range(n_levels):
                # unique unit names
                cur_names_unique = np.unique(self.units[:, ii])
                seq_idxs = []
                for name in cur_names_unique:
                    # find sequences of unit names
                    seq_idxs.append(findseq(self.units[:, ii], name))
                seq_idxs = np.concatenate(seq_idxs, axis=0)

                # create boxes and label with unit names
                for jj, name in enumerate(cur_names_unique):
                    # box
                    cur_x = -unit_label_wid_tot + ii * unit_label_wid
                    cur_y = self.base_height[seq_idxs[jj, 0]]
                    cur_orig = (cur_x, cur_y)
                    cur_height = self.top_height[seq_idxs[jj, 1]] - cur_y
                    cur_rect = Rectangle(cur_orig, unit_label_wid, cur_height,
                                         edgecolor='k', facecolor=(1, 1, 1, 0))
                    ax.add_patch(cur_rect)
                    # label
                    cur_x_text = cur_x + unit_label_wid/2
                    cur_y_text = cur_y + cur_height/2
                    # print(f'{cur_x_text}, {name}')
                    cur_text = ax.text(cur_x_text, cur_y_text, name, 
                            va='center_baseline', ha='center', rotation_mode='anchor',
                            rotation=90, fontsize=unit_fontsize)
                    plt.gcf().canvas.draw()
                    # get resulting data coordinates of the plotted label
                    transf = ax.transData.inverted()
                    bb = cur_text.get_window_extent()
                    bb_datacoords = bb.transformed(transf)
                    # if overstepping the box, give a warning and remove
                    if (bb_datacoords.y0 < cur_y) | (bb_datacoords.y1 > cur_y + cur_height):
                        print(f'Unit label {name} too big for box, removing.')
                        cur_text.remove()

        # prettify
        # ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        for label in ax.get_xticklabels():
            label.set_rotation(270)
            label.set_ha('center')
            label.set_va('top')
        ax.set_axisbelow(True)
        ax.xaxis.grid(ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Height (m)')

        # turning the spines off creates some clipping mask issues
        # so just turn the clipping masks off
        for obj in ax.findobj():
            obj.set_clip_on(False)

    def style_compatibility(self, style):
        """
        Check that section is compatible with a `Style`. 
        Returns True if so, False if not

        Parameters
        ----------
        section : Section
            A Section object.

        style : Style
            A Style object.
        """
        # assert that all facies in the section are present in style
        if not np.all(np.in1d(self.facies, style.labels)):
            missing_style = self.facies[~np.in1d(self.facies, style.labels)]
            warnings.warn(f'{missing_style} in {self.name} not in style.')
            return False
        # check that annotations, if present, are in style
        elif (self.annotations is not None) and (style.annotations is not None):
            if not np.all(np.in1d(self.annotations.annotation, list(style.annotations.keys()))):
                missing_ann = self.annotations.annotation[~np.in1d(self.annotations.annotation, list(style.annotations.keys()))].values
                warnings.warn(f'{missing_ann} not in style.')
                return False
            else:
                return True 
        else:
            return True

    def add_facies_attribute(self, attribute_name, attribute_values):
        """
        Add an attribute that is explicitly tied to the stratigraphic
        units.

        This function should only be used to add an attribute which
        corresponds 1:1 with each measured unit. Typically, such
        attributes would add additional detail (e.g. grain size or
        bedforms) to the broad facies description of the unit.

        To add an attribute that is tied to the stratigraphic height,
        but not explicitly tied to the stratigraphic units (e.g.
        chemostratigraphic data), use the Data subclass via the method
        `add_data_attribute()`.

        To add an attribute that is neither tied to the stratigraphic
        height nor the stratigraphic units (e.g. the GPS coordinates of
        the start and end of the section), use
        `add_generic_attribute()`.

        Parameters
        ----------
        attribute_name : string
            The name of the attribute.

        attribute_values : 1d array_like
            The attribute values. NaNs are accepted.
        """
        # convert to arrays and check the dimensionality
        attribute_values = attribute_convert_and_check(attribute_values)

        # check that the length of the attribute data matches the number of units
        n_attribute_units = len(attribute_values)
        if n_attribute_units != self.n_beds:
            raise Exception('Length of thicknesses and attribute data are not '
                            'equal.')

        # assign the data to the object
        setattr(self, attribute_name, attribute_values)

        # keep track of the attribute
        facies_attributes = self.facies_attributes
        facies_attributes.append(attribute_name)
        setattr(self, 'facies_attributes', facies_attributes)

    def return_facies_dataframe(self):
        """
        Return a Pandas DataFrame of all attributes associated with the
        stratigraphic units.

        Parameters
        ----------
        None.

        Returns
        -------
        df : DataFrame
            Pandas DataFrame with all attributes associated with the
            stratigraphic units.
        """
        df = pd.DataFrame({'unit_number': np.arange(self.n_beds)})

        for attribute in self.facies_attributes:
            df[attribute] = getattr(self, attribute)

        return df

    def add_generic_attribute(self, attribute_name, attribute_values):
        """
        Add an attribute that is neither tied to the stratigraphic
        height nor the stratigraphic units.

        To add an attribute that is explicitly tied to the stratigraphic
        units, use `add_facies_attribute()`.

        To add an attribute that is tied to the stratigraphic height,
        but not explicitly tied to the stratigraphic units (e.g.
        chemostratigraphic data), use the Data subclass via the method
        `add_data_attribute()`.

        Parameters
        ----------
        attribute_name : string
            The name of the attribute.

        attribute_values : any
            The attribute values.
        """
        # convert from pandas series to arrays if necessary
        if type(attribute_values) == pd.core.series.Series:
            attribute_values = attribute.values

        # assign the data to the object
        setattr(self, attribute_name, attribute_values)

        # keep track of the attribute
        generic_attributes = self.generic_attributes
        generic_attributes.append(attribute_name)
        setattr(self, 'generic_attributes', generic_attributes)

    def add_data_attribute(self, attribute_name, attribute_height,
                           attribute_values):
        """
        Add an attribute that is tied to the stratigraphic height, but
        not explicitly tied to the stratigraphic units.

        A typical example of such data would be chemostratigraphic data.

        Parameters
        ----------
        attribute_name : string
            The name of the attribute.

        attribute_height : 1d array_like
            The stratigraphic heights at which the attribute were
            generated.

        attribute_values : 1d array_like
            The attribute values.
        """
        # if the attribute already exists, remove it and replace it
        if hasattr(self, attribute_name):
            delattr(self, attribute_name)
            # remove from the list of data attributes
            self.data_attributes.remove(attribute_name)

        setattr(self, attribute_name,
                self.Data(attribute_height, attribute_values))

        # keep track of the attribute
        data_attributes = self.data_attributes
        data_attributes.append(attribute_name)
        setattr(self, 'data_attributes', data_attributes)

    def get_units(self, heights):
        """Return unit(s) at requested height(s). Units must be defined. If multiple
        units are defined, all are returned.

        Parameters
        ----------
        heights : float or array_like
            Height(s) at which to query units

        Returns
        -------
        units : array-like
            Unit(s) at queried height(s)
        """

        if self.units is None:
            warnings.warn('No units defined.')
            return None

        heights = np.atleast_1d(heights)
        unit_idx = (heights.reshape(-1, 1) < self.top_height) & \
                   (heights.reshape(-1, 1) >= (self.base_height))

        units = np.tile(self.units, (len(heights), 1, 1))[unit_idx]

        units = np.squeeze(units)

        return units

    class Data:
        """
        This nested class stores any data tied to the stratigraphic
        height, but not explicitly tied to the stratigraphic units.

        A typical example of such data would be chemostratigraphic data.

        Parameters
        ----------
        attribute_height : 1d array_like
            The stratigraphic heights at which the attribute were
            generated.

        attribute_values : 1d array_like
            The attribute values.

        Attributes
        ----------
        height : 1d array_like
            The stratigraphic heights at which the attribute were
            generated.
        
        values : 1d array_like
            The attribute values.

        n_values : int
            The number of values in the attribute.
        """

        def __init__(self, attribute_height, attribute_values):
            """
            Initialize Data with the two primary attributes.
            """
            # convert to arrays and check the dimensionality
            attribute_height = attribute_convert_and_check(attribute_height)
            attribute_values = attribute_convert_and_check(attribute_values)

            # check that the heights are numeric
            # if attribute_height.dtype == np.object:
            #     raise Exception('Height data must be floats or ints.')

            # check that the X and Y have the same length
            n_height = len(attribute_height)
            n_values = len(attribute_values)
            if n_height != n_values:
                raise Exception('Length of X and Y data are not equal.')

            # assign the attributes
            self.height = attribute_height
            self.values = attribute_values

            # add some other useful attributes
            self.n_values = n_values

            # keep track of attributes
            self.height_attributes = ['height', 'values']
            self.generic_attributes = ['n_values']

        def add_height_attribute(self, attribute_name, attribute_values):
            """
            Add an attribute that is tied to the stratigraphic heights
            of this instance of Data.

            A typical example would be the sample names associated with
            a given chemostratigraphic profile.

            Parameters
            ----------
            attribute_name : string
                The name of the attribute.

            attribute_values : 1d array_like
                The attribute values. NaNs are accepted.
            """
            # convert to arrays and check the dimensionality
            attribute_values = attribute_convert_and_check(attribute_values)

            # check that the length of the attribute data matches the number of values
            n_attribute_units = len(attribute_values)
            if n_attribute_units != self.n_values:
                raise Exception('Length of heights and attribute data are not '
                                'equal.')

            # assign the data to the object
            setattr(self, attribute_name, attribute_values)

            # keep track of the attribute
            height_attributes = self.height_attributes
            height_attributes.append(attribute_name)
            setattr(self, 'height_attributes', height_attributes)

        def return_data_dataframe(self):
            """
            Return a Pandas DataFrame of all attributes associated with
            this Data object.

            Parameters
            ----------
            None.

            Returns
            -------
            df : DataFrame
                Pandas DataFrame with all attributes associated with
                this Data object,
            """
            df = pd.DataFrame({'height': self.height})

            for attribute in self.height_attributes:
                df[attribute] = getattr(self, attribute)

            return df

        def add_data_facies(self, section):
            """
            .. deprecated:: 2.0
                Functionality will be changed in future versions.

            Extract the facies associated with each data point.

            Parameters
            ----------
            section : Section
                The parent Section object for this Data object.
                (While the parent Section object is already implied for any
                given Data object, accessing the Section object's attributes
                is not elegant, so we pass it as an argument for now...)

            Notes
            -----
            Samples marked with a .5 in the 'unit' column come from unit boundaries.
            User input is required to assign the correct unit.

            If the sample comes from the lower unit, subtract 0.5. If the sample comes
            from the upper unit, add 0.5.
            """
            warnings.warn('add_data_facies() is deprecated.', DeprecationWarning, stacklevel=2)
            # make sure the data points are sorted
            if not np.array_equal(np.sort(self.height), self.height):
                raise Exception(
                    'Stratigraphic heights were out of order. '
                    'Please check that the data were not logged incorrectly, '
                    'and sort by stratigraphic height before using '
                    'add_data_attribute().')

            # calculate the unit from which the sample came
            facies = []
            unit_number = []
            for i in range(len(self.height)):
                unit_ind = 0

                # stop once the base_height is equal to or greater than the sample height
                # (with some tolerance for computer weirdness)
                while section.top_height[unit_ind] < (self.height[i] - 1e-5):
                    unit_ind = unit_ind + 1

                # if the sample comes from a unit boundary, mark with .5
                # (with some tolerance for computer weirdness)
                if abs(section.top_height[unit_ind] - self.height[i]) < 1e-5:
                    unit_number.append(unit_ind + 0.5)
                    facies.append('!!!ON BOUNDARY!!!')
                else:
                    unit_number.append(unit_ind)
                    facies.append(section.facies[unit_ind])

            # assign the data to the object
            setattr(self, 'facies', facies)
            setattr(self, 'unit_number', unit_number)

            # keep track of the attributes
            height_attributes = self.height_attributes
            height_attributes.append('facies')
            height_attributes.append('unit_number')
            setattr(self, 'height_attributes', height_attributes)

        def clean_data_facies_helper(self, data_name):
            """
            .. deprecated:: 2.0
                Functionality will be changed in future versions.

            Prints the code for cleaning up data points on unit boundaries.

            Parameters
            ----------
            data_name : string
                Variable name used for the Data object (e.g. 'section_01.data_01').

            Notes
            -----
            Copy and paste the printed code into a cell and edit as follows:

            If the sample comes from the lower unit, subtract 0.5. If the sample comes
            from the upper unit, add 0.5

            After fixing all changes, copy and paste the dataframe into the data .csv.
            """
            warnings.warn('clean_data_facies_helper() is deprecated.', DeprecationWarning, stacklevel=2)

            # check that add_data_facies has been run
            if 'facies' not in self.height_attributes:
                raise Exception(
                    'Data has not been assigned a facies. '
                    'Data.add_data_facies() will assign facies and '
                    'unit numbers automatically.')
            if 'unit_number' not in self.height_attributes:
                raise Exception(
                    'Data has not been assigned a unit_number. '
                    'Data.add_data_facies() will assign facies and '
                    'unit numbers automatically.')

            # get the dataframe
            df = self.return_data_dataframe()

            # get the samples on unit boundaries
            mask = df['unit_number'] != np.floor(df['unit_number'])
            df_slice = df[mask]

            # the case where there are no samples on units boundaries
            if len(df_slice.index) == 0:
                print(
                    'No samples are on unit boundaries - no manual edits required.'
                )

            # print the code
            else:
                print(
                    '1) Copy and paste the code below into a cell and edit as follows:'
                )
                print(
                    '- If the sample comes from the lower unit, subtract 0.5.')
                print('- If the sample comes from the upper unit, add 0.5.')
                print('')
                print('2) Run Data.clean_data_facies().')
                print('===')
                for i in df_slice.index:
                    print(data_name + '.unit_number[' + str(i) + '] = ' +
                          str(df_slice['unit_number'][i]) + ' #height = ' +
                          str(df_slice['height'][i]) + ', sample = ' +
                          str(df_slice['sample'][i]))

        def clean_data_facies(self, section):
            """
            .. deprecated:: 2.0
                Functionality will be changed in future versions.

            Extract the facies associated with each data point AFTER cleaning up
            the samples that are from unit boundaries using Data.clean_data_facies_helper().

            Parameters
            ----------
            section : Section
                The parent Section object for this Data object.
                (While the parent Section object is already implied for any
                given Data object, accessing the Section object's attributes
                is not elegant, so we pass it as an argument for now...)
            """
            warnings.warn('clean_data_facies() is deprecated.', DeprecationWarning, stacklevel=2)

            # convert unit_number to int
            self.unit_number = np.array(self.unit_number, dtype=int)

            # reassign facies
            for i in range(len(self.unit_number)):
                self.facies[i] = section.facies[self.unit_number[i]]


class Style():
    """Section plotting style class.

    Organizes the plotting style for the lithostratigraphy, including colors, width, swatches, and annotations.

    Parameters
    ----------
    labels : 1d array_like
        The labels to which colors and widths are assigned. When plotting a
        Section, values within the facies of that Section
        must form a subset of the values within this array_like.
    color_values : array_like
        The colors that will be assigned to the associated labels.
        Values must be interpretable by matplotlib.
    width_values : 1d array_like of floats
        The widths that will be assigned to the associated labels.
        Values must be between 0 and 1.
    swatch_values : 1d array-like of int
        USGS swatch codes (see swatches/png/) for labels.
        Give zero for no swatch.
    annotations : list, dict, or None
        Specification of annotations to plot alongside sections, by default None. 
        If None, no annotations are plotted. 
        User can also provide a list of annotation names to select among the default annotations to plot. See :py:meth:`Style.plot_default_annotations()` for the default annotations provided by pystrat.
        Alternatively, the user can provide a dictionary linking annotation names to png file paths for plotting custom annotations.
    swatch_wid : float (default 1.5)
        Width of the swatch pattern in inches.
    """

    def __init__(self,
                 labels,
                 color_values,
                 width_values,
                 swatch_values=None,
                 annotations=None,
                 swatch_wid=1.5):
        """Initialize Style
        """
        # convert to arrays and check the dimensionality
        labels = attribute_convert_and_check(labels)
        width_values = attribute_convert_and_check(width_values)

        if swatch_values is not None:
            swatch_values = np.asarray(swatch_values).astype(int).ravel().tolist()

        # convert from pandas series/dataframes to arrays if necessary
        if type(color_values) == pd.core.series.Series:
            color_values = color_values.values
        if type(color_values) == pd.core.frame.DataFrame:
            color_values = color_values.values

        # check that the widths are between 0 and 1
        if np.max(width_values) > 1 or np.min(width_values) < 0:
            raise Exception('Width values must be floats between 0 and 1.')

        # assign the attributes
        self.style_attribute = 'facies'
        self.labels = labels
        self.color_values = color_values
        self.width_values = width_values
        self.swatch_values = swatch_values
        self.swatch_wid = swatch_wid

        # validate annotations
        if annotations is not None:
            if isinstance(annotations, list):
                # check that all annotations are in the default set
                default_annotations_pngs = os.listdir(resources.files('pystrat').joinpath('annotations'))
                default_annotations = [annotation_png.split('.')[0] for annotation_png in default_annotations_pngs]
                for annotation in annotations:
                    if annotation not in default_annotations:
                        warnings.warn(f'Annotation {annotation} not in default set and will not be plotted.')
                        annotations.remove(annotation)
                # make into dictionary with paths to default pngs
                annotations = {annotation: resources.files('pystrat').joinpath('annotations', f'{annotation}.png') for annotation in annotations}
            # if annotations are a dictionary
            elif isinstance(annotations, dict):
                # check that all annotations are png files
                for annotation in annotations:
                    if not os.path.isfile(annotations[annotation]):
                        warnings.warn(f'Annotation {annotation} is not a png file. No annotation will be plotted.')
                        del annotations[annotation]
            else:
                raise ValueError('Annotations must be a list or a dictionary.')
        self.annotations = annotations

        # add some other useful attributes
        self.n_labels = len(labels)

    @staticmethod
    def plot_default_annotations(ax=None):
        """Plot annotations provided by pystrat.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes, optional
            Axes to plot into, by default None. If None, creates an axis.

        Examples
        --------
        .. plot::
            :include-source:

            import pystrat
            pystrat.Style.plot_default_annotations()
        """
        if ax is None:
            ax = plt.axes()

        # retrieve default annotations
        annotation_pngs = os.listdir(resources.files('pystrat').joinpath('annotations'))
        annotations = [annotation_png.split('.')[0] for annotation_png in annotation_pngs]

        ax.set_ylim([0, len(annotation_pngs)])
        ax.autoscale(False)

        # plot annotations
        for ii, annotation_png in enumerate(annotation_pngs):
            cur_path = resources.files('pystrat').joinpath('annotations', annotation_png)
            pos = [0, ii]
            plot_annotation(cur_path, pos, 0.5, ax)

        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks(0.3 + np.arange(len(annotation_pngs)))
        ax.set_yticklabels(annotations, fontsize=10)
        ax.tick_params(axis='y', left=False)

    def plot_legend(self, 
                    ax=None, 
                    legend_unit_height=0.25,
                    fontsize=10,
                    annotations_loc='bottom'):
        """
        Plot a legend for this Style object.

        Parameters
        ----------
        legend_unit_height : float
            A scaling factor to modify the height of each unit in the
            legend only.
        ax : matplotlib.Axes, optional
            Axis to plot into, defaults to None. If None, creates an axis.
        fontsize : float (default 10)
            Fontsize for text in the legend.
        annotations_loc : str (default 'right')
            Location of the annotations, default is 'bottom'. Options are 'right', 'bottom', or 'top'.

        Returns
        -------
        fig : matplotlib Figure
            Figure handle.
        ax : matplotlib Axes
            Axis handle.
        """

        # extract attributes
        labels = self.labels
        color_values = self.color_values
        width_values = self.width_values
        swatch_values = self.swatch_values

        # sort by width
        width_sort_inds = np.argsort(width_values)
        labels = labels[width_sort_inds]
        color_values = color_values[width_sort_inds]
        width_values = width_values[width_sort_inds]

        # what does this do??
        if swatch_values is not None:
            swatch_values = [swatch_values[x] for x in width_sort_inds]

        # initiate ax
        if ax == None:
            ax = plt.axes()

        # determine the axis height and limits first
        ax_height = legend_unit_height * self.n_labels
        ax.set_ylim(0, self.n_labels)
        ax.set_xlim([0, 1])

        # initiate counting of the stratigraphic height
        strat_height = 0

        # loop over each item
        for i in range(len(labels)):

            # create the rectangle - with thickness of 1
            ax.add_patch(
                Rectangle((0.0, strat_height),
                          width_values[i],
                          1,
                          facecolor=color_values[i],
                          edgecolor='k'))

            # if swatch is defined, plot it
            # if swatch_values[0] != None:
            if swatch_values is not None:
                if swatch_values[i] != 0:
                    extent = [
                        0, width_values[i], strat_height, strat_height + 1
                    ]
                    plot_swatch(swatch_values[i], extent, ax,
                                swatch_wid=self.swatch_wid)

            # label the unit
            # ax.text(-0.01,
            #         strat_height + 0.5,
            #         labels[i],
            #         horizontalalignment='right',
            #         verticalalignment='center')

            # count the stratigraphic height
            strat_height = strat_height + 1

        # plot annotations 
        if self.annotations is not None:
            n_annotations = len(self.annotations)

            if annotations_loc == 'right':
                # plot each one beside the column
                for ii in range(n_annotations):
                    height = 0.6
                    pos = [1.1, ii+0.5-height/2]

                    # for text, assume min aspect ratio of image of 0.5
                    width = height/0.5*get_axis_aspect(ax)
                    ax.text(1.1+width, ii+0.5, list(self.annotations)[ii], 
                            va='center',
                            fontsize=fontsize)
                    plot_annotation(list(self.annotations.values())[ii], pos, height, ax)
            
            elif annotations_loc == 'bottom':
                # plot each one below the column
                for ii in range(n_annotations):
                    height = 0.6
                    pos = [0.0, (-ii-0.5) -height/2]

                    # for text, assume min aspect ratio of image of 0.5
                    width = height/0.5*get_axis_aspect(ax)
                    ax.text(0.0 - width/2, -(ii+0.5), list(self.annotations)[ii], 
                            va='center',
                            ha='right',
                            fontsize=fontsize)
                    plot_annotation(list(self.annotations.values())[ii], pos, height, ax)

            elif annotations_loc == 'top':
                # plot each one below the column
                for ii in range(n_annotations):
                    height = 0.6
                    pos = [0.0, (ii + 0.5) -height/2 + self.n_labels]

                    # for text, assume min aspect ratio of image of 0.5
                    width = height/0.5*get_axis_aspect(ax)
                    ax.text(0.0 - width/2, (ii+0.5) + self.n_labels, list(self.annotations)[ii], 
                            va='center',
                            ha='right',
                            fontsize=fontsize)
                    plot_annotation(list(self.annotations.values())[ii], pos, height, ax)

        # prettify
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        # ax.xaxis.grid(ls='--')
        # ax.set_yticklabels([])
        # ax.set_yticks([])
        ax.set_yticks(np.arange(len(labels))+0.5)
        ax.set_yticklabels(labels, fontsize=fontsize)
        ax.tick_params(axis='y', width=0, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # turning the spines off creates some clipping mask issues
        # so just turn the clipping masks off
        for obj in ax.findobj():
            obj.set_clip_on(False)


########################
### HELPER FUNCTIONS ###
########################

def attribute_convert_and_check(attribute):
    """
    Convert pandas series to arrays (if necessary) and check that the
    data are 1d.

    This function assists the addition of attributes to Section and
    Data.

    Parameters
    ----------
    attribute : 1d array_like
        The attribute to be added.

    Returns
    -------
    attribute : 1d array
        The attribute after the conversion and check.
    """
    # convert from pandas series to arrays if necessary
    if type(attribute) == pd.core.series.Series:
        attribute = attribute.values

    # make at least 1d
    attribute = np.atleast_1d(attribute)
    # check that the data are 1d
    if attribute.ndim != 1:
        raise Exception('Data must be 1d.')

    return attribute


def plot_swatch(swatch_code, extent, ax, swatch_wid=1.5, warn_size=False):
    """Plot a tesselated USGS geologic swatch to fit a desired extent

    Parameters
    ----------  
    swatch_code : int
        USGS swatch code

    extent : 1d array_like
        rectangular area in which to tesselate the swatch [x0, x1, y0, y1]

    ax : matplotlib axis
        axis in which to plot

    swatch_wid : float
        width of original swatch image file in inches

    warn_size : boolean (default: False)
        Whether or not to issue warnings on swatch sizes.
    """

    x0, x1, y0, y1 = extent

    # dimensions of extent (data coordinates)
    dx_ex = x1 - x0
    dy_ex = y1 - y0

    if dx_ex == 0 or dy_ex == 0:
        if warn_size:
            warnings.warn(
                'Extent has no width and/or height. Swatch cannot be plotted.')
        return

    # load swatch, use swatch code as file name
    swatch_code = int(swatch_code) # make sure integer
    swatch_path = resources.files('pystrat') / 'swatches' / f'{swatch_code}.png'
    # try to open swatch
    try:
        swatch = Image.open(swatch_path)
    except FileNotFoundError:
        raise ValueError(f'Swatch {swatch_code} not found.')
   
    ax_x_in, ax_y_in = get_inch_per_dat(ax)

    dx_ex_in = dx_ex * ax_x_in
    dy_ex_in = dy_ex * ax_y_in

    # size of image
    dx_sw, dy_sw = swatch.size  # in pixels
    asp_sw = dy_sw / dx_sw
    dx_sw_in = swatch_wid
    dy_sw_in = asp_sw * dx_sw_in

    # tile image to overlap with extent
    sw_np = np.array(swatch)
    ny_tile = np.ceil(dy_ex_in / dy_sw_in).astype(int)
    nx_tile = np.ceil(dx_ex_in / dx_sw_in).astype(int)
    sw_tess = np.tile(sw_np, [ny_tile, nx_tile, 1])

    # now crop tessalated image to fit
    x_idx_crop = int(dx_ex_in / dx_sw_in / nx_tile * sw_tess.shape[1])
    y_idx_crop = int(dy_ex_in / dy_sw_in / ny_tile * sw_tess.shape[0])
    sw_tess = sw_tess[0:y_idx_crop, 0:x_idx_crop, :]

    if sw_tess.shape[0] == 0 or sw_tess.shape[1] == 0:
        if warn_size:
            warnings.warn(
                'Extent has no width and/or height. Swatch cannot be plotted.')
        return

    ax.imshow(sw_tess, extent=extent, zorder=2, aspect='auto')
    # ax.imshow(sw_tess, extent=extent, zorder=2)
    ax.autoscale(False)


def plot_annotation(annotation_path, pos, height, ax,):
    """Plot a png of an annotation.

    Parameters
    ----------  
    annotation_path : str
        Path to the file of the annotation

    pos : 1d array_like
        left and bottom of the annotation plotting, data units [x0, y0]

    height : numeric
        vertical thickness to plot (vertical extent will be y0 to y0 + height)

    ax : matplotlib axis
        axis in which to plot

    """

    # os agnostic path
    annotation_path = os.path.normpath(annotation_path)

    # try to load the annotation image
    try:
        annotation = Image.open(annotation_path)
    except FileNotFoundError:
        warnings.warn(f'Annotation {annotation_path} not found.')
        return

    # convert to numpy array
    ann_arr = np.array(annotation)
    
    # axis inches per data unit
    aspect_axis = get_axis_aspect(ax)

    # size of image
    dx, dy= annotation.size  # in pixels
    aspect_ann = dy / dx
    width = height/aspect_ann*aspect_axis

    extent = [pos[0], pos[0]+width, pos[1], pos[1]+height]

    ax.imshow(ann_arr, extent=extent, zorder=2, aspect='auto')
    ax.autoscale(False)


def get_inch_per_dat(ax):
    """Returns the inches per data unit of the given axis 
    (x, y)

    Parameters
    ----------
    ax : axis handle

    Returns
    -------
    x_inch_per_dat : float
        inches per data unit in x
    
    y_inch_per_dat : float
        inches per data unit in y
    """
    fig = ax.get_figure()
    # check for SubFigures
    while type(fig).__name__ != 'Figure':
        fig = fig.get_figure()

    figsize = fig.get_size_inches()

    x_inch_per_dat = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (1, 0)])), axis=0)[0][0] * \
                         figsize[0]
    y_inch_per_dat = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (0, 1)])), axis=0)[0][1] * \
                         figsize[1]
    return x_inch_per_dat, y_inch_per_dat


def get_axis_aspect(ax):
    """Returns the aspect ratio of the axis in terms of inches per data unit for each axis

    Parameters
    ----------
    ax : axis handle

    Returns
    -------
    aspect : float
        aspect ratio of the axis
    """
    x_inch_per_dat, y_inch_per_dat = get_inch_per_dat(ax)
    return y_inch_per_dat/x_inch_per_dat



###
### HELPER FUNCTIONS
###
def findseq(x, val, noteq=False):
    """
    Find sequences of a given value within an input vector.

    Parameters
    ----------
    x : array_like
        Vector of values in which to find sequences.
    val : float
        Value to find sequences of in x.
    noteq : bool, optional
        Whether to find sequences equal or not equal to the supplied value.
        Default is False.
     
    Returns
    -------
    idx : array
        Array that contains in rows the number of total sequences of val, with the first column containing the begin indices of each sequence, the second column containing the end indices of sequences, and the third column contains the length of the sequence.
    """
    x = x.copy().squeeze()
    assert len(x.shape) == 1, "x must be vector"
    # indices of value in x, and
    # compute differences of x, since subsequent occurences of val in x will
    # produce zeros after differencing. append nonzero value at end to make
    # x and difx the same size
    if noteq:
        validx = np.atleast_1d(np.argwhere(x != val).squeeze())
        logidx = x != val
        x[validx] = val+1
        difx = np.append(np.diff(logidx),1)
    else:
        validx = np.atleast_1d(np.argwhere(x == val).squeeze())
        logidx = x == val
        # difx = np.append(np.diff(x),1)
        difx = np.append(np.diff(logidx),1)
    nval = len(validx)
    # if val not in x, warn user
    if nval == 0:
        warnings.warn("value val not found in x")
        return 0

    # now, where validx is one and difx is zero, we know that we have
    # neighboring values of val in x. Where validx is one and difx is nonzero,
    # we have end of a sequence

    # now loop over all occurrences of val in x and construct idx
    c1 = 0
    idx = np.empty((1,3))
    while c1 < nval:
        curidx = np.array([[validx[c1],validx[c1],1]])
        c2 = 0
        while difx[validx[c1]+c2] == 0:
            curidx[0,1] += 1
            curidx[0,2] += 1
            c2 += 1
        idx = np.append(idx,curidx,axis=0)
        c1 = c1+c2+1
    idx = idx[1:,:].astype(int)
    return idx