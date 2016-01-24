import seaborn as sns

sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook', font_scale=1.5)

def add_spt_axis(axis, spt_values=('M5', 'M0', 'K5', 'K0', 'G5', 'G0')):
    from kglib.spectral_type import SpectralTypeRelations
    MS = SpectralTypeRelations.MainSequence()
    # Find the temperatures at each spectral type
    temp_values = MS.Interpolate('Temperature', spt_values)
    
    # make the axis
    top = axis.twiny()
    
    # Set the full range to be the same as the data axis
    xlim = axis.get_xlim()
    top.set_xlim(xlim)
    
    # Set the ticks at the temperatures corresponding to the right spectral type
    top.set_xticks(temp_values)
    top.set_xticklabels(spt_values)
    top.set_xlabel('Spectral Type')
    return top


def bokeh_errorbar(fig, x, y, xerr=None, yerr=None, source=None, color='blue', point_kwargs=None, error_kwargs=None):
    if point_kwargs is None:
        point_kwargs = {}
    if error_kwargs is None:
        error_kwargs = {}

    if source is not None:
        df = source.to_df()
        x = df[x].values
        y = df[y].values
        xerr = df[xerr].values if xerr is not None else xerr
        yerr = df[yerr].values if yerr is not None else yerr
        if 'source' not in point_kwargs:
            point_kwargs['source'] = source

    if 'color' not in point_kwargs:
        point_kwargs['color'] = color
    if 'color' not in error_kwargs:
        error_kwargs['color'] = color

    fig.circle(x, y, **point_kwargs)

    if xerr is not None:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        fig.multi_line(x_err_x, x_err_y, **error_kwargs)

    if yerr is not None:
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        fig.multi_line(y_err_x, y_err_y, **error_kwargs)
