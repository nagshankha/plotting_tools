from empty_class import EmptyClass
import numpy as np
import pandas as pd
import warnings

class PlottingTools(EmptyClass):

   def __init__(self, rcParams=None, backend='tkagg'):
      self.preamble(rcParams, backend)
      self.assign_color_palette()
      
   def preamble(self, rcParams, backend):
   
      import matplotlib as mpl
      mpl.use(backend)
      import matplotlib.pyplot as plt
            
      mpl.rcParams['axes.labelpad'] = 15
      mpl.rcParams['legend.fontsize'] = 'medium'
      mpl.rcParams['lines.markersize'] = 15
      mpl.rcParams['xtick.major.size'] = 8
      mpl.rcParams['xtick.major.size'] = 10.0
      mpl.rcParams['xtick.minor.size'] = 6.0
      mpl.rcParams['xtick.major.width'] = 2.0
      mpl.rcParams['xtick.minor.width'] = 1.5
      mpl.rcParams['xtick.major.pad'] = 10
      mpl.rcParams['xtick.minor.pad'] = 10
      mpl.rcParams['lines.linewidth'] = 1.5
      mpl.rcParams['patch.linewidth'] = 1.5
      mpl.rcParams['axes.linewidth'] = 1.5
      mpl.rcParams['ytick.major.size'] = 8
      mpl.rcParams['ytick.major.size'] = 10.0
      mpl.rcParams['ytick.minor.size'] = 6.0
      mpl.rcParams['ytick.major.width'] = 2.0
      mpl.rcParams['ytick.minor.width'] = 1.5
      mpl.rcParams['ytick.major.pad'] = 10
      mpl.rcParams['ytick.minor.pad'] = 10
      mpl.rcParams['ytick.right'] = True
      mpl.rcParams['xtick.direction'] = 'in'
      mpl.rcParams['ytick.direction'] = 'in'
      mpl.rcParams['font.size'] = 30
      mpl.rcParams['font.family']='sans-serif'
      mpl.rcParams['font.sans-serif']=['Helvetica']
      #mpl.rcParams['font.weight']=500
      #mpl.rcParams['axes.labelweight']=500
      mpl.rcParams['axes.axisbelow'] = True
      mpl.rcParams['axes.xmargin'] = 0.1
      #from matplotlib.legend_handler import HandlerTuple
      #from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
      #                               AutoMinorLocator, LogFormatter)
      
      if rcParams is None:
         pass
      elif not isinstance(rcParams, dict):
         raise ValueError('rcParams must be a dictionary')
      else:
         for x in rcParams:
            mpl.rcParams[x] = rcParams[x]
      
      rc_params = {#'pdf.fonttype': 42,
                   'text.usetex' : True,
                   'lines.antialiased' : True,
                   'ps.papersize'      : 'A4',
                   'ps.usedistiller'   : 'xpdf',
                   'text.latex.preamble' : (
                                            r'\usepackage[T1]{fontenc}' +
                                            r'\usepackage{cmbright}' +
                                            #r'\usepackage{mathptmx}'+
                                            #r'\usepackage{helvet}'+
                                            #r'\usepackage{courier}'+
                                            #r'\usepackage{times}' +
                                            #r'\usepackage{txfonts}' +
                                            r'\usepackage{siunitx}' +
                                            r'\usepackage{xcolor}' +
                                            r'\usepackage{graphicx}' 
                                           )
                  }
      plt.rcParams.update(rc_params) 
      self.plt = plt; self.mpl = mpl
      
   def assign_color_palette(self):
      
      # color reference: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
      colors = dict([('Black',       [0.0, 0.0, 0.0]), #Black
                     ('Red',         [255, 0.0, 0.0]), #100% Red
                     ('Green',       [0.0, 255, 0.0]), #100% Green
                     ('Blue',        [0.0, 0.0, 255]), #100% Blue
                     ('Maroon',      [128, 0.0, 0.0]), #Maroon
                     ('Brown',       [170, 110, 40.]), #Brown
                     ('Olive',       [128, 128, 0.0]), #Olive
                     ('Teal',        [0.0, 128, 128]), #Teal
                     ('Navy',        [0.0, 0.0, 128]), #Navy
                     ('Light Red',   [230, 25.0, 75]), #Light Red
                     ('Orange',      [245, 130, 48.]), #Orange
                     ('Yellow',      [255, 255, 25.]), #Yellow
                     ('Lime',        [210, 245, 60.]), #Lime
                     ('Another Green', [60, 180, 75.0]), #Another Green
                     ('Cyan',        [70, 240, 240.]), #Cyan
                     ('Light Blue',  [0.0, 130, 200]), #Light Blue
                     ('Purple',      [145, 30, 180.]), #Purple
                     ('Magenta',     [240, 50, 230.]), #Magenta
                     ('Grey',        [128, 128, 128]), #Grey
                     ('Pink',        [250, 190, 190]), #Pink
                     ('Apricot',     [255, 125, 180]), #Apricot
                     ('Mint',        [170, 255, 195]), #Mint
                     ('Beige',       [255, 250, 200]), #Beige (Very light; preferably exclude)
                     ('Lavender',    [230, 190, 255]) #Lavender
                     ])
                         
   
      self.color_palette = pd.DataFrame.from_dict(colors)/255.
      
   def show_color_palette(self):
      
      fig = self.plot_colortable(pd.DataFrame.to_dict(self.color_palette,
                                                      orient='list'), 
               "Colors easily distinguishable to human eye",  
               sort_colors=False, emptycols=0)
      fig.show()
      
   def plot_colortable(self, colors, title, sort_colors=True, emptycols=0):
      
      #**Reference: https://matplotlib.org/stable/gallery/color/named_colors.html
      
      import matplotlib.colors as mcolors
      from matplotlib.patches import Rectangle

      cell_width = 212
      cell_height = 22
      swatch_width = 48
      margin = 12
      topmargin = 40
   
      # Sort colors by hue, saturation, value and name.
      if sort_colors is True:
         by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                                name)
                          for name, color in colors.items())
         names = [name for hsv, name in by_hsv]
      else:
         names = list(colors)
   
      n = len(names)
      ncols = 4 - emptycols
      nrows = n // ncols + int(n % ncols > 0)
   
      width = cell_width * 4 + 2 * margin
      height = cell_height * nrows + margin + topmargin
      dpi = 72
   
      fig, ax = self.plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
      fig.subplots_adjust(margin/width, margin/height,
                          (width-margin)/width, (height-topmargin)/height)
      ax.set_xlim(0, cell_width * 4)
      ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
      ax.yaxis.set_visible(False)
      ax.xaxis.set_visible(False)
      ax.set_axis_off()
      ax.set_title(title, fontsize=24, loc="left", pad=10)
   
      for i, name in enumerate(names):
         row = i % nrows
         col = i // nrows
         y = row * cell_height
   
         swatch_start_x = cell_width * col
         text_pos_x = cell_width * col + swatch_width + 7
   
         ax.text(text_pos_x, y, name, fontsize=14,
                 horizontalalignment='left',
                 verticalalignment='center')
   
         ax.add_patch( Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                       height=18, facecolor=colors[name], edgecolor='0.7')
                     )
   
      return fig
      
   def create_figure(self, **kwargs):
      
      self.fig = self.plt.figure(**kwargs)
   
   def add_gridspec(self, gs_name, nrows=1, ncols=1, **kwargs):
      
      self.__dict__[gs_name] = self.fig.add_gridspec(nrows, ncols, **kwargs)
      
   def add_subgridspec(self, sub_gs_name, parent_gs_name, nrows, ncols, 
                       gs_span0_tuple=(0,1), gs_span1_tuple=(0,1), **kwargs):
      
      if not hasattr(self, parent_gs_name):
         raise AttributeError('"self" does not have the "{0}" attribute'.format(
                                parent_gs_name))
      
      self.__dict__[sub_gs_name] = self.__dict__[parent_gs_name][
                           gs_span0_tuple[0]:gs_span0_tuple[1],
                           gs_span1_tuple[0]:gs_span1_tuple[1]].subgridspec(
                                 nrows, ncols, **kwargs)
      
   def visualize_gridspec(self, gs_name):
      
      ms = np.meshgrid(*[np.arange(x) for x in self.__dict__[gs_name].get_geometry()])
      tups = list(zip(ms[0].flatten(), ms[1].flatten()))
      for t1, t2 in tups:
         ax = self.fig.add_subplot(self.__dict__[gs_name][t1, t2])
         ax.annotate('{0}[{1},{2}]'.format(gs_name, t1, t2), xy=(0.5, 0.5), 
                      xycoords='axes fraction', va='center', ha='center')
      self.show()
      
   def add_axis(self, axis_name, gs_name, gs_span0_tuple=(0,1), gs_span1_tuple=(0,1), 
                **kwargs):
      
      self.__dict__[axis_name] = self.fig.add_subplot(self.__dict__[gs_name][
                                          gs_span0_tuple[0]:gs_span0_tuple[1],
                                          gs_span1_tuple[0]:gs_span1_tuple[1]],
                                          **kwargs)
   
   def add_twin_axis(self, axis_name_to_twin, ax_twin='x', spline_shift=1.0):
      
      if not hasattr(self, axis_name_to_twin):
         raise ValueError('The axis requested to twin does not exist')

      if axis_name_to_twin[-7:-1] in ['_twinx', '_twiny']:
         raise RuntimeError('It is not possible to twin an already twinned axis '+
                            'with this method')
      largest_twin_index = 0   
      if ax_twin == 'x':
         for att in self.__dict__:
            if att[:-1] == axis_name_to_twin+'_twinx':
               largest_twin_index = max(largest_twin_index, int(att[-1]))
         if largest_twin_index == 3:
            raise RuntimeError('It is not possible to have more than three '+
                               'twinx axes for any given axis with this method')
         else:
            largest_twin_index += 1
            new_twin_axis_name = axis_name_to_twin+f'_twinx{largest_twin_index}'
            self.__dict__[new_twin_axis_name] = self.__dict__[
                                           axis_name_to_twin].twinx()
            if spline_shift >= 1.0:
               self.__dict__[new_twin_axis_name].spines.right.set_position(
                                                      ("axes", spline_shift))
            elif spline_shift < 0.0:
               self.__dict__[new_twin_axis_name].spines.left.set_position(
                                                      ("axes", spline_shift))
            else:
               raise ValueError('spline_shift must be <0 or >=1')
      elif ax_twin == 'y':
         for att in self.__dict__:
            if att[:-1] == axis_name_to_twin+'_twiny':
               largest_twin_index = max(largest_twin_index, int(att[-1]))
         if largest_twin_index == 3:
            raise RuntimeError('It is not possible to have more than three '+
                               'twiny axes for any given axis with this method')
         else:
            largest_twin_index += 1
            new_twin_axis_name = axis_name_to_twin+f'_twiny{largest_twin_index}'
            self.__dict__[new_twin_axis_name] = self.__dict__[
                                           axis_name_to_twin].twiny()
            if spline_shift >= 1.0:
               self.__dict__[new_twin_axis_name].spines.top.set_position(
                                                      ("axes", spline_shift))
            elif spline_shift < 0.0:
               self.__dict__[new_twin_axis_name].spines.bottom.set_position(
                                                      ("axes", spline_shift))
            else:
               raise ValueError('spline_shift must be <0 or >=1')
      else:
         raise ValueError('The axis spline to twin "ax_twin" must be either '+
                          '"x" or "y"')
      
   def annotate(self, axis_name, annotation_text, xy=(0.5, 0.5),
                xycoords='axes fraction', va='bottom', ha='left', 
                color='Black', **kwargs):
      
      self.__dict__[axis_name].annotate(annotation_text, xy=xy, xycoords=xycoords,
                                        va=va, ha=ha, color=self.color_palette[color],
                                        **kwargs)
      
   def add_axis_labels(self, axis_name, xlabel=None, ylabel=None, 
                       **text_props):
      
      if 'xlabel_props' not in text_props:
         text_props['xlabel_props'] = {}
         
      if 'ylabel_props' not in text_props:
         text_props['ylabel_props'] = {}
      
      if xlabel is not None:
         if isinstance(xlabel, str):
            if 'label_position' in text_props['xlabel_props']:
               self.__dict__[axis_name].xaxis.set_label_position(
                                 text_props['xlabel_props'].pop('label_position'))
            self.__dict__[axis_name].set_xlabel(xlabel, **text_props['xlabel_props'])
         else:
            raise ValueError('"xlabel" must be a string')
            
      if ylabel is not None:
         if isinstance(ylabel, str):
            if 'label_position' in text_props['ylabel_props']:
               self.__dict__[axis_name].yaxis.set_label_position(
                                 text_props['ylabel_props'].pop('label_position'))
            self.__dict__[axis_name].set_ylabel(ylabel, **text_props['ylabel_props'])
         else:
            raise ValueError('"ylabel" must be a string')
            
   def add_axis_title(self, axis_name, title=None):
       
      if title is not None:
         if isinstance(title, str):
            self.__dict__[axis_name].set_title(title)
         else:
            raise ValueError('Axis "title" must be a string')
      
   
   def plot_pandas(self, axis_name, data, plot_handle_name='h_plot', **kwargs):
      
      if not isinstance(data, (pd.DataFrame, pd.Series)):
         raise TypeError('The data to be plotted must be a pandas DataFrame '+
                         'or Series')
      else:
         self.__dict__[plot_handle_name], = data.plot(ax=self.__dict__[axis_name],
                                                      **kwargs)
         
         
   def add_legend(self, axis_name, handle_names, labels, **kwargs):
      
      handles = [self.__dict__[x] for x in handle_names]
      l = self.__dict__[axis_name].legend(handles, labels, **kwargs)
      return l
      
   def modify_x_ticklabels(self, axis_name, new_labels, tick_locs = None,
                             rotation=None, colors = None):
      
      if tick_locs is None:
         tick_locs = self.__dict__[axis_name].get_xticks()
         
      self.__dict__[axis_name].set_xticks(tick_locs, labels=new_labels)
      self.__dict__[axis_name].tick_params(axis='x', rotation=rotation)
      if isinstance(colors, list) and set(colors).issubset(set(self.color_palette.columns)):
         for xtick, color in zip(self.__dict__[axis_name].get_xticklabels(), 
                                 self.color_palette[colors].values.T):
            xtick.set_color(color)
      
   def unify_axes_lims(self, axes_to_unify, unify_with, axis='both'):
      
      if isinstance(unify_with, self.mpl.axes._base._AxesBase):
         x_ax_lims_to_set = unify_with.get_xlim()
         y_ax_lims_to_set = unify_with.get_ylim()
      else:
         raise ValueError('"unify_with" must be a matplotlib Axis instance')
      
      if isinstance(axes_to_unify, self.mpl.axes._base._AxesBase):
         if axis == 'both':
            axes_to_unify.set_xlim(x_ax_lims_to_set)
            axes_to_unify.set_ylim(y_ax_lims_to_set)
         elif axis == 'x':
            axes_to_unify.set_xlim(x_ax_lims_to_set)
         elif axis == 'y':
            axes_to_unify.set_ylim(y_ax_lims_to_set)
         else:
            raise ValueError('axis must be either x or y or both')
      elif isinstance(axes_to_unify, list):
         if len(axes_to_unify) == 0:
            return
         else:
            self.unify_axes_lims(axes_to_unify.pop(), unify_with, axis=axis)
            self.unify_axes_lims(axes_to_unify, unify_with, axis=axis)
      else:
         raise ValueError('"axes_to_unify" must either be a matplotlib Axis '+
                          'instance or it must be a list of matplotlib Axis '+
                          'instances.')
         
   def combine_axes_lims(self, axes_to_combine_lims:list, axis='both'):
      
      if not isinstance(axes_to_combine_lims, list):
         raise ValueError('"axes_to_combine_lims" must be a list')
      elif not np.all([isinstance(self.__dict__[x], self.mpl.axes._base._AxesBase) 
                       for x in axes_to_combine_lims]):
         raise ValueError('All list entries in "axes_to_combine_lims" must be '+
                          'matplotlib Axis instances')
      else:
         x_lims_arr = np.c_[[self.__dict__[x].get_xlim() for x in axes_to_combine_lims]].T
         y_lims_arr = np.c_[[self.__dict__[x].get_ylim() for x in axes_to_combine_lims]].T
         
         for x in axes_to_combine_lims:
            if axis == 'both':
               self.__dict__[x].set_xlim((np.min(x_lims_arr[0]), np.max(x_lims_arr[1])))
               self.__dict__[x].set_ylim((np.min(y_lims_arr[0]), np.max(y_lims_arr[1])))
            elif axis == 'x':
               self.__dict__[x].set_xlim((np.min(x_lims_arr[0]), np.max(x_lims_arr[1])))
            elif axis == 'y':
               self.__dict__[x].set_ylim((np.min(y_lims_arr[0]), np.max(y_lims_arr[1])))
            else:
               raise ValueError('axis must be either x or y or both')
            
         
   def show(self):
      
      self.plt.show()   
      
      
   def __setattr__(self, name, value):
            
      if name == 'fig':
         self.__dict__[name] = value
         
      self.__dict__[name] = value
      
      
################################################################################
         
class Plot2DLinear(PlottingTools):
   
   def __init__(self, y=None, x=None, axes={None:None}, fig=None, **kwargs):
      
      if 'rcParams' not in kwargs:
         PlottingTools.__init__(self)
      else:
         PlottingTools.__init__(self, kwargs['rcParams'])
      self.add_data(x, y)
      self.add_axes(axes, fig)
      
   def add_data(self, x, y):
      
      if y is None:
         self.y0 = np.arange(10)
      else:
         self.y0 = y
         
      if x is None:
         self.x0 = np.arange(len(self.y0))
      else:
         self.x0 = x
      
      if len(self.y0) != len(self.x0):
         raise ValueError('x and y must have same number of datapoints, that is '+
                          'the same length')
         
   def add_axes(self, axes, fig):
      
      if axes != {None:None}:
         self.axes = axes
      elif fig is not None:
         self.fig = fig
      else:
         warnings.warn('There are no axes yet. Axes must be initiated before '+
                       'plotting')
         
   def analyze_data(self, y_n_quartiles=1, conf_level=0.95):
      
      if not isinstance(y_n_quartiles, int):
         raise TypeError('y_n_quartiles must be an integer')
      elif y_n_quartiles < 0:
         raise ValueError('y_n_quartitles must be non-negative')
      else:
         pass
      
      import scipy.stats
      
      if self.x0.ndim == 2:
         self.n_x_data = np.shape(self.x0)[1]
         if self.n_x_data > 30:
            Z_x = scipy.stats.norm.ppf(1 - (0.5*(1-conf_level)))
         else:
            Z_x = scipy.stats.t.ppf(1 - (0.5*(1-conf_level)), self.n_x_data-1)
         
      if self.y0.ndim == 2:
         self.n_y_data = np.shape(self.y0)[1]
         if self.n_y_data > 30:
            Z_y = scipy.stats.norm.ppf(1 - (0.5*(1-conf_level)))
         else:
            Z_y = scipy.stats.t.ppf(1 - (0.5*(1-conf_level)), self.n_y_data-1)
      
      if self.x0.ndim == 1:
         self.__dict__['x'] = self.x0
      else:
         self.__dict__['x'] = np.mean(self.x0, axis=1)
         self.__dict__['x_std'] = np.std(self.x0, axis=1)*Z_x/np.sqrt(self.n_x_data)
         
      if self.y0.ndim == 1:
         self.__dict__['y'] = self.y0
      else:
         self.__dict__['y'] = np.mean(self.y0, axis=1)
         self.__dict__['y_std'] = np.std(self.y0, axis=1)*Z_y/np.sqrt(self.n_y_data)
         self.__dict__['y_quartile_0'] = (np.c_[np.min(self.y0, axis=1),
                                               np.max(self.y0, axis=1)] )
         if y_n_quartiles > 0:
            self.__dict__['y0'] = np.sort(self.y0, axis=1)
            for i in range(1, y_n_quartiles+1):
               q = i*0.5/(y_n_quartiles+1)
               self.__dict__[f'y_quartile_{i}'] = (self.y0[:, 
                            [int(q*np.shape(self.y0)[1])-1, 
                             int((1-q)*np.shape(self.y0)[1])-1]] )
      
         
   def plot_continuous_data(self, axis_name, plot_handle_name=None,
                            uncertainty_band=False, uncertainty_band_steps = 1,
                            std_err_bars_x = False, std_err_bars_y = False,
                            color = 'Black', color_twin_axes=True, 
                            errorbar_parameters={}, **kwargs):
      
      if not isinstance(uncertainty_band_steps, int):
         raise TypeError('Uncertainty band steps must always be an integer')
      elif uncertainty_band_steps%2 == 0:
         raise ValueError('Uncertainty band steps must be an odd integer')
      elif uncertainty_band_steps < 0:
         raise ValueError('Uncertainty band steps must be positive')
      else:
         y_n_quartiles = int((uncertainty_band_steps-1)/2)
         
      self.analyze_data(y_n_quartiles = y_n_quartiles)   
      
      if plot_handle_name is None:
         if color is None:
            self.__dict__[axis_name].plot(self.x, self.y, **kwargs)
         elif color not in self.color_palette:
            self.__dict__[axis_name].plot(
                                          self.x, self.y, color=color,
                                          **kwargs)
         else:
            self.__dict__[axis_name].plot(self.x, self.y, 
                                          color=self.color_palette[color], 
                                          **kwargs)
      else:
         if isinstance(plot_handle_name, str):
            if color is None:
               self.__dict__[plot_handle_name], = self.__dict__[axis_name].plot(
                                             self.x, self.y,  **kwargs)
            elif color not in self.color_palette:
               self.__dict__[plot_handle_name], = self.__dict__[axis_name].plot(
                                             self.x, self.y, color=color,
                                             **kwargs)
            else:
               self.__dict__[plot_handle_name], = self.__dict__[axis_name].plot(
                                                self.x, self.y, 
                                                color=self.color_palette[color],
                                                **kwargs)
         else:
            raise ValueError('Plot handle name must be a string')
      
      if self.x0.ndim == 1 and std_err_bars_x:
         raise ValueError('Do not ask for x errorbars when there is a single '+
                          'entry for every x point')
      if self.y0.ndim == 1 and any([uncertainty_band, std_err_bars_y]):
         raise ValueError('Do not ask for y errorbars and/or uncertainty band '+
                          ' when there is a single entry for every y point')        
      
      if std_err_bars_x:
         xerr = self.x_std
      else:
         xerr = None
      if std_err_bars_y:
         yerr = self.y_std
      else:
         yerr = None
         
      if set(errorbar_parameters.keys()).issubset({'elinewidth', 'capsize',
                                                   'barsabove', 'lolims',
                                                   'uplims', 'xlolims',
                                                   'xuplims', 'errorevery',
                                                   'capthick'}):
         pass
      else:
         raise ValueError('errorbar_parameters must be a dict with keys from '+
                          "the following: {'elinewidth', 'capsize', 'barsabove', "+
                          "'lolims', 'uplims', 'xlolims', 'xuplims', "+ 
                          "'errorevery', 'capthick'}")
      
      
      if std_err_bars_x or std_err_bars_y:
         if plot_handle_name is None:
            if color is None:
               self.__dict__[axis_name].errorbar(self.x, self.y, yerr=yerr, xerr=xerr, 
                                   fmt='none', **errorbar_parameters)
            elif color not in self.color_palette:
               self.__dict__[axis_name].errorbar(self.x, self.y, yerr=yerr, xerr=xerr, 
                                   fmt='none', ecolor=color, **errorbar_parameters)
            else:
               self.__dict__[axis_name].errorbar(self.x, self.y, yerr=yerr, xerr=xerr, 
                                   fmt='none', ecolor=self.color_palette[color],
                                   **errorbar_parameters)
         else:
            if color is None:
               self.__dict__[plot_handle_name+'_errbar_container'] = self.__dict__[axis_name].errorbar(self.x, self.y, yerr=yerr, xerr=xerr, 
                                   fmt='none', **errorbar_parameters)
            elif color not in self.color_palette:
               self.__dict__[plot_handle_name+'_errbar_container'] = self.__dict__[axis_name].errorbar(self.x, self.y, yerr=yerr, xerr=xerr, 
                                   fmt='none', ecolor=color, **errorbar_parameters)
            else:
               self.__dict__[plot_handle_name+'_errbar_container'] = self.__dict__[
               axis_name].errorbar(self.x, self.y, yerr=yerr, xerr=xerr, 
                                   fmt='none', ecolor=self.color_palette[color],
                                   **errorbar_parameters)
                                   
      if uncertainty_band:
         if y_n_quartiles<5:
            alpha_arr = np.linspace(0.1, 0.5, 5)[-(y_n_quartiles+1):]
         else:
            alpha_arr = np.linspace(0.1, 0.5, (y_n_quartiles+1))
         for i in range(y_n_quartiles+1):
            if i == y_n_quartiles:
               if plot_handle_name is None:
                  self.__dict__[axis_name].fill_between(self.x, 
                                       self.__dict__[f'y_quartile_{i}'][:,0],
                                       self.__dict__[f'y_quartile_{i}'][:,1],
                                       facecolor=self.color_palette[color],
                                       alpha = alpha_arr[i])
               else:
                  self.__dict__[plot_handle_name+f'_uncertainty_band_step_{i+1}'] = ( 
                                self.__dict__[axis_name].fill_between(self.x, 
                                       self.__dict__[f'y_quartile_{i}'][:,0],
                                       self.__dict__[f'y_quartile_{i}'][:,1],
                                       facecolor=self.color_palette[color],
                                       alpha = alpha_arr[i]) )
                  
            else:
               if plot_handle_name is None:
                  self.__dict__[axis_name].fill_between(self.x,
                                       self.__dict__[f'y_quartile_{i}'][:,0],
                                       self.__dict__[f'y_quartile_{i+1}'][:,0],
                                       facecolor=self.color_palette[color],
                                       alpha = alpha_arr[i])
                  self.__dict__[axis_name].fill_between(self.x,
                                       self.__dict__[f'y_quartile_{i}'][:,1],
                                       self.__dict__[f'y_quartile_{i+1}'][:,1],
                                       facecolor=self.color_palette[color],
                                       alpha = alpha_arr[i])
               else:
                  self.__dict__[plot_handle_name+f'_uncertainty_band_step_{i+1}'] = ( 
                                self.__dict__[axis_name].fill_between(self.x, 
                                       self.__dict__[f'y_quartile_{i}'][:,0],
                                       self.__dict__[f'y_quartile_{i+1}'][:,0],
                                       facecolor=self.color_palette[color],
                                       alpha = alpha_arr[i]) )
                  self.__dict__[plot_handle_name+'_uncertainty_band_step'+
                                f'_{uncertainty_band_steps-i}'] = ( 
                                self.__dict__[axis_name].fill_between(self.x, 
                                       self.__dict__[f'y_quartile_{i}'][:,1],
                                       self.__dict__[f'y_quartile_{i+1}'][:,1],
                                       facecolor=self.color_palette[color],
                                       alpha = alpha_arr[i]) )          
      
      if color_twin_axes:
         if hasattr(self, axis_name+'_twinx'):
            self.__dict__[axis_name].yaxis.label.set_color(
                                   self.__dict__[plot_handle_name].get_color())
            self.__dict__[axis_name].tick_params(axis='y', which='both',
                         colors=list(self.__dict__[plot_handle_name].get_color()))
            self.__dict__[axis_name].spines.right.set_visible(False)
            self.__dict__[axis_name].spines.left.set_color(
               self.__dict__[plot_handle_name].get_color())
         elif hasattr(self, axis_name+'_twiny'):
            self.__dict__[axis_name].xaxis.label.set_color(
                                   self.__dict__[plot_handle_name].get_color())
            self.__dict__[axis_name].tick_params(axis='x', which='both',
                         colors=list(self.__dict__[plot_handle_name].get_color()))
            self.__dict__[axis_name].spines.top.set_visible(False)
            self.__dict__[axis_name].spines.bottom.set_color(
               self.__dict__[plot_handle_name].get_color())            
         if axis_name[-6:] == '_twinx':
            self.__dict__[axis_name].yaxis.label.set_color(
                                   self.__dict__[plot_handle_name].get_color())
            self.__dict__[axis_name].tick_params(axis='y', which='both',
                         colors=list(self.__dict__[plot_handle_name].get_color()))
            self.__dict__[axis_name].spines.left.set_visible(False)
            self.__dict__[axis_name].spines.right.set_color(
               self.__dict__[plot_handle_name].get_color())
         elif axis_name[-6:] == '_twiny':
            self.__dict__[axis_name].xaxis.label.set_color(
                                   self.__dict__[plot_handle_name].get_color())
            self.__dict__[axis_name].tick_params(axis='x', which='both',
                         colors=list(self.__dict__[plot_handle_name].get_color()))
            self.__dict__[axis_name].spines.bottom.set_visible(False)
            self.__dict__[axis_name].spines.top.set_color(
               self.__dict__[plot_handle_name].get_color())       
            

            
   def __setattr__(self, name, value):
            
      PlottingTools.__setattr__(self, name, value)
      
      if name in ['x0', 'y0']:
         if not isinstance(value, np.ndarray):
            raise TypeError('{0} must be a numpy ndarray'.format(name))
         elif value.ndim > 2 or value.ndim == 0:
            raise ValueError('{0} must be either a 1D or 2D array'.format(name))
         elif not np.issubdtype(value.dtype, np.number):
            raise TypeError('x and y must be numeric arrays.')
         else:
            self.__dict__[name] = value
            
      elif name == 'axes':
         import matplotlib
         if not isinstance(value, dict):
            raise ValueError('axes must be a dictionary')
         elif not ( np.all([isinstance(x, str) for x in value.keys()]) and 
                    np.all([isinstance(x, matplotlib.axes._base._AxesBase) 
                            for x in value.values()]) ):
            raise ValueError('The keys in axes are axis names which must be strings '+
                             'and the values are corresponding matplotlib Axes instances.')
         else:
            for x in value:
               self.__dict__[x] = value[x]
            print(', '.join(value.keys) + ' axes are created.')
               
      elif name == 'fig':
         import matplotlib
         if isinstance(value, matplotlib.figure.Figure):
            for i, x in enumerate(value.axes):
               self.__dict__[f'axis{i+1}'] = x
            print(', '.join([f'axis{i+1}' for i in range(len(value.axes))]) + 
                  ' axes are created.')
         else:
            raise ValueError('fig must be a matplotlib Figure instance')
               
      
            
            
      #self.__dict__[name] = value
         
         
      
      
   
      
      
      
      
      
      
      
