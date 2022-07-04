"""
Created on Tue Jul  6 21:04:40 2021

@authors: Sam Dawley // Tristan Kooistra

conda environment configuration
python 3.7 interpreter or greater
dependencies:
numpy
scipy
matplotlib
pandas
openpyxl

"""

import getpass
import pandas as pd
import numpy as np
import scipy.spatial as spatial
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
import math
from scipy.spatial import Delaunay

def make_voronoi(file):
    """
    Parameters
    ----------
    file : EXCEL SPREADSHEET. File name must include (x,y) coords 
    in first 2 columns WITHOUT a header.
    Also, put file in folder '/Users/user/Raw Voronoi Data/XXX' where 'XXX' is file name WITHOUT '.xlsx'

    Returns
    -------
    Complete colored voronoi diagram
    """
    
    # Import data from excel file and convert to numpy array
    data = pd.read_excel(r"{0}/{1}.xlsx".format(file_path(), file), header=None)
    data.to_numpy()
    coords = np.array(data)
    xs = coords[:, 0]
    ys = coords[:, 1]
    center = ((max(xs)+min(xs))/2, (max(ys)+min(ys))/2)
    
    # Extract Delaunay triangulation and convex hull from data set to be
    # used later for masking 
    delaunay: Delaunay = spatial.Delaunay(data)
    hull = spatial.ConvexHull(data)
    
    #Find minimum cell to cell distance
    if min_dists:
        min_cell_to_cell = np.zeros(len(delaunay.points))
        for ii, vv in enumerate(delaunay.points):
            all_simp_verts = find_neighbors(ii, delaunay)
            #pbar2.update(ii+1)
            dists = []
            for ind_vert in all_simp_verts:
                pt_simp = delaunay.points[ind_vert]
                pt_pt = delaunay.points[ii]
                dist = np.linalg.norm(pt_pt - pt_simp)
                dists.append(dist/scale)
                min_cell_to_cell[ii] = min(dists)
        df = pd.DataFrame(min_cell_to_cell)
        df.to_excel("{0}/{1}_minDist.xlsx".format(results_path, file), header=False, index=False)
    
    # Uses voronoi_volumes function to compute the areas of all cells 
    # made in the voronoi diagram. Appends all areas to a list and 
    # finds the maximum and minimum of the list to be used for the 
    # normalization of the color map. Notice that maxima is changed
    # by a scalar to make the coloring nicer
    vols, indices = voronoi_volumes(data)
    areas = []
    for element in vols:
        if element != np.inf or element != 0:
            areas.append(element)
    areas2 = np.array(areas)
    lookout = find_max(areas2)
    minima = min(areas2)
    maxima = max_value_scale_factor*lookout
    df = pd.DataFrame(areas2)
    df.to_excel("{0}/{1}_areas.xlsx".format(results_path, file), header=False, index=False)
    
    # Normalizing the color map of the diagram
    norm = color_norm_type(
        vmin=minima,
        vmax=maxima,
        clip=True
        )
    mapper = matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=color_map
        )
    
    # Normalizing the color map of the color bar 
    color_bar_norm = matplotlib.colors.Normalize(
        vmin=100,
        vmax=maxima,
        clip=True
        )
    color_bar_mapper = matplotlib.cm.ScalarMappable(
        norm=color_bar_norm,
        cmap=color_map
        )
    color_bar_mapper.set_clim(min(areas2)-1000, find_max(areas2)+1000)
    
    # Creating the voronoi figure
    fig, ax = plt.subplots()
    vor = spatial.Voronoi(
        data,
        furthest_site = False,
        ) 
    regions_st, vertices_st = voronoi_finite_polygons_2d(vor)

    figure_v = spatial.voronoi_plot_2d(
        vor,
        ax = ax,
        show_vertices=False,
        show_points=True,
        s=1,
        line_colors=line_color_plus_background_color,
        line_width=line_thickness,
        point_size=0,
        )
    figure_v.set_figwidth(figure_dimensions[0])
    figure_v.set_figheight(figure_dimensions[1])
    
    plt.tick_params(left=False, bottom=False)
    plt.title(plot_title)
    
    # Calculating areas of voronoi cells and coloring them
    # with the color map defined above
    polygons = []
    for reg in regions_st:
        polygon = vertices_st[reg]
        polygons.append(polygon)
    for r in range(len(vor.point_region)):
       region = vor.regions[vor.point_region[r]]
       if not -1 in region:
           polygon = [vor.vertices[i] for i in region]
           area_p = shoelace(polygon)
           plt.fill(*zip(*polygon), clip_on=True, color=mapper.to_rgba(area_p))
    
    # Creating path on ConvexHull
    hull_vertices = []
    hull_codes = [mpath.Path.MOVETO]
    for n in hull.vertices:
        hull_vertices.append(hull.points[n])
        hull_codes.append(mpath.Path.CURVE4)
    hull_codes.remove(mpath.Path.CURVE4)
    hull_codes.remove(mpath.Path.CURVE4)
    hull_codes.append(mpath.Path.CLOSEPOLY)
    
    # Creating path on circle
    circ_path = matplotlib.path.Path.circle(
        center,
        radius=center[0]*circ_scale
        )
    circ_path_codes = [mpath.Path.MOVETO]
    circ_path_codes.extend([mpath.Path.CURVE4 for n in range(len(circ_path.vertices)-2)])
    circ_path_codes.append(mpath.Path.CLOSEPOLY)
        
    # Creating path on rectangle surrounding entire figure
    rect_vertices = [(-1000, -1000), (9000, -1000), (9000, 9000), (-1000, 9000)]
    rect_codes = [mpath.Path.LINETO for p in range(len(rect_vertices))]
    rect_codes[0] = mpath.Path.MOVETO
    
    # Concatenating list of hull vertices and rectangle vertices
    verts_all = []
    if convex:
        verts_all.extend(hull_vertices)
    else:
        verts_all.extend(circ_path.vertices)
    verts_all.extend(rect_vertices[::-1])
    # Concatenating list of hull codes and rectangle codes
    codes_all = []
    if convex:
        codes_all.extend(hull_codes)
    else:
        codes_all.extend(circ_path_codes)
    codes_all.extend(rect_codes)
    
    # Creating paths from vertices and codes made above
    path_all = matplotlib.path.Path(verts_all, codes_all)

    # Creating patches from paths above. Patches will act as masks
    patch_all = mpatches.PathPatch(
        path_all,
        fc=line_color_plus_background_color,
        ec=line_color_plus_background_color
        )
    plt.gca().add_patch(patch_all)
    
    # Making color bar 
    if color_bar_visibility:
        cbar = plt.colorbar(color_bar_mapper, shrink=0.75)
        cbar.set_ticks([p*find_max(areas2)/6 for p in range(7)])
        if color_bar_ticks == "data":
            cbar_labels_d = [""]
            cbar_labels_d.extend(["$10^{0}$ \u03BCm\u00b2".format(OrderOfMagnitude(find_max(areas2)/10**p)) for p in reversed(range(5))])
            cbar_labels_d.append("")
        elif color_bar_ticks == "fixed":
            cbar_labels_f = [""]
            cbar_labels_f.extend(["$10^{0}$ \u03BCm\u00b2".format(p) for p in range(2, 7)])
            cbar_labels_f.append("")
            cbar.set_ticklabels(cbar_labels_f)
        cbar.ax.tick_params(labelsize=font_size, length=25, color="white")
        cbar.ax.xaxis.set_tick_params(pad=20)
        cbar.set_label("", labelpad=+80, size=50, rotation=270)

    # Setting axis parameters
    ax.axis(axis_visibility)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax_labels = ["{0} \u03BCm".format(1000*m) for m in range(7)] # Labeling tick labels at every 1000 microns
    ax_labels.append("")
    
    # Major tick parameters
    ax.tick_params(
        which="major",
        direction="inout",
        length=30,
        width=8,
        color="k",
        pad=15
        )
    # Minor tick parameters
    ax.tick_params(
        which="minor",
        direction="inout",
        left=True,
        bottom=True,
        length=20,
        width=4,
        color="k",
        )
    for axs in ['bottom', 'left']:
        ax.spines[axs].set_linewidth(8)
        ax.tick_params(
            left=True,
            bottom=True,
            labelbottom=True,
            labelleft=True
            )
        
    # x-Axis specific parameters
    ax.spines["bottom"].set_visible(x_axis_visibility)
    if x_axis_visibility:
        ax.spines["bottom"].set_bounds(0, max(xs))
        ax_marks_x = [p*max(xs)/6.5 for p in range(7)] # Making tick marks at even intervals 
        ax_marks_x.append(max(xs)) # Appending max value tick mark to axis 
        plt.xticks(ax_marks_x, ax_labels, fontsize=font_size)
        plt.xlabel(x_axis_label, fontsize=font_size, labelpad=50)
    else:
        ax.tick_params(axis="x", color="w", labelbottom=False)
        
    # y-Axis specific parameters
    ax.spines["left"].set_visible(y_axis_visibility)
    if y_axis_visibility:
        ax.spines["left"].set_bounds(0, max(ys))
        ax_marks_y = [0]
        ax_marks_y.extend([(p-0.5)*max(ys)/6.5 for p in range(1,8)]) # Making tick marks at even intervals
        plt.yticks(ax_marks_y[::-1], ax_labels, fontsize=font_size)
        plt.ylabel(y_axis_label, fontsize=font_size, labelpad=50)
    else:
        ax.tick_params(axis="y", color="w", labelleft=False)
        
    # Minor tick mark placement
    minor_ticks_x = [p*max(xs)/6.5*(2*p-1)/(2*p) for p in range(1,7)]
    minor_ticks_x.append(max(xs))
    minor_ticks_y = [p*max(ys)/6.5 for p in range(7)]
    minor_ticks_y.append(max(ys))
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks_x))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks_y[::-1]))

    # Finishing diagram and plotting
    plt.gca().invert_yaxis()
    plt.savefig("{0}/{1}_voronoi.pdf".format(fig_path, file))
    print("Plot finished.")
    return
    
# Parameters which are easily changed
color_map = matplotlib.cm.RdPu_r # Link to available color maps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
                                 # Tristan likes 'RdPu_r'
line_color_plus_background_color = "white"
line_thickness = 1
max_value_scale_factor = 0.5 # Anywhere from 0.01-1 seems to work well
color_norm_type = matplotlib.colors.LogNorm # 'LogNorm' or 'Normalize' work the best
x_axis_label = "Distance"
y_axis_label = "Distance"
plot_title = ""
font_size = 50
axis_visibility = "on"
x_axis_visibility = True
y_axis_visibility = True
color_bar_visibility = True # If 'True' make sure figure_dimensions is 5x4 ratio
color_bar_ticks = "fixed" # 'fixed' or 'data'
figure_dimensions = (60, 48) # Width x height, 5x4 is a good ratio if color_bar_visibility = True
scale = 1.1292 # Pixel/Micron ratio
min_dists = False # True --> find min distances. False --> don't
convex = False # True --> use convex hull to clip image. Else, use circular clip
circ_scale = 0.95 # Factor by which to change the circular clip size. Used if convex = False
