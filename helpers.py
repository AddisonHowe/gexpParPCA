import numpy as np
import matplotlib.pyplot as plt

def get_bin_ranges(ncells, bin_sz, overlap):
    bin_ranges = [[x, x+bin_sz] for x in range(0, ncells, bin_sz - overlap)]
    bin_ranges = [p for p in bin_ranges if p[1] <= ncells]
    bin_ranges[-1][1] = ncells
    return bin_ranges

def get_bins_from_intervals(srt, bin_ranges, bidx0, bidx1):
    pst_bins = [srt[brng[0]:brng[1]] for brng in bin_ranges[bidx0 : bidx1 + 1]]
    return [np.sort(x) for x in pst_bins]

def load_psts(pst_fname):
    """
    Given path to pseudotime file, assumes a header, and separated values "index pseudotime".
    Returns numpy 2d array of floats [[idx1, pst1], [idx2, pst2], ...]
    """
    return np.genfromtxt(pst_fname, skip_header=True, dtype='float')
    
def val2cellcolor(x):
    """Red scale color map"""
    return (1,1-x,1-x)
    
def eigenvector_loading_barchart(idx_subset, evec_subset, gnames, eps=0.01, 
                                 do_table=True, verbose=False, title=None, saveas=None, cmap=None, dpi=150):

    # Get indices of genes for which square of loading is greater than epsilon
    interesting_gene_idxs = np.unique(np.where(evec_subset**2 > eps)[1])
    interesting_gene_names = gnames[interesting_gene_idxs]
    num_int_genes = len(interesting_gene_idxs)
    if verbose:
        print(f"Interesting Genes ({num_int_genes}):\n{interesting_gene_names}")

    # Collect data (squared loadings) for interesting genes. Recall must sum to 1.
    data = np.zeros([num_int_genes+1, len(idx_subset)])
    for i, idx in enumerate(idx_subset):
        evec = evec_subset[i]
        data[:-1,i] = evec[interesting_gene_idxs]**2
        data[-1,i] = 1 - np.sum(data[:-1,i])  # all other loadings
    
    columns = idx_subset
    rows = list(interesting_gene_names) + ['Other']

    # Get colors
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(rows)))
    np.random.shuffle(colors)
    if cmap:
        colormap = {}
        for i, rowname in enumerate(rows):
            if rowname in cmap:
                colormap[rowname] = cmap[rowname]
                colors[i] = cmap[rowname]
            else:
                colormap[rowname] = colors[i]
        colormap = {**cmap, **colormap}
    else: 
        colormap = {rows[i]: colors[i] for i in range(len(rows))}
    
    n_rows = len(data)

    index = np.arange(0,1,1/len(columns)) + 1/len(columns)/2
    bar_width = 0.50 * 1/len(columns)

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    fig, ax = plt.subplots(1, 1, figsize=(24,18), dpi=dpi)

    # Plot bars and create text labels for the table
    cell_text = []
    cell_colors = []
    for row in range(n_rows):
        ax.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        if do_table:
            cell_text.append([f'{x:.2g}' for x in data[row]])
            cell_colors.append([val2cellcolor(x) for x in data[row]])
    
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    if do_table:
        cell_text.reverse()
        cell_colors.reverse()
        rows.reverse()

    # Add a table at the bottom of the axes
    if do_table:
        the_table = ax.table(cellText=cell_text,
                             rowLabels=rows,
                             rowColours=[[*r[0:-1], 0.75] for r in colors],
                             colLabels=columns,
                             loc='bottom', 
                             cellColours=cell_colors
                            )
    ax.set_xlim([0,1])
    ax.set_ylabel("Square comp")
    if title:
        ax.set_title(title)
    
    if do_table:
        ax.set_xticks([])
        the_table.scale(1, 1.5)
        cellDict = the_table.get_celld()
#         the_table.auto_set_font_size(False)
#         print(cellDict[(0,0)].get_height())
#         the_table.set_fontsize(cellDict[(0,0)].get_height())
        
    else:
        ax.set_xticks(index[0::5])
        ax.set_xticklabels(idx_subset[0::5])
    
    if saveas:
        plt.savefig(saveas, bbox_inches='tight')

    plt.show()
                     
    return data, interesting_gene_idxs, interesting_gene_names, colormap