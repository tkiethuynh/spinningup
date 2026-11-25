import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import sys
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def _is_interactive_backend():
    """Return True if Matplotlib can open interactive GUI windows."""
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, font_scale=1.5, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value], dtype=np.float64)
            finite = np.isfinite(x)
            x_safe = np.where(finite, x, 0.0)
            counts = np.convolve(finite.astype(np.float64), y, 'same')
            smoothed_x = np.where(
                counts > 0,
                np.convolve(x_safe, y, 'same') / np.maximum(counts, 1.0),
                np.nan,
            )
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set_theme(style="darkgrid", font_scale=font_scale)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, errorbar='sd', **kwargs)
    legend = plt.legend(loc='best')
    if legend is not None and _is_interactive_backend():
        legend.set_draggable(True)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                with open(os.path.join(root, 'config.json')) as config_file:
                    config = json.load(config_file)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except Exception:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except Exception:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets


def _collect_logdirs_and_datasets(all_logdirs, legend=None, select=None, exclude=None):
    if not all_logdirs:
        raise ValueError("No log directories specified.")

    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir) or '.'
            if not osp.isdir(basedir):
                continue
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x and osp.isdir(fulldir(x))])

    logdirs = list(dict.fromkeys(logdirs))

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data, logdirs


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    data, _ = _collect_logdirs_and_datasets(all_logdirs, legend, select, exclude)
    return data


def _resolve_savefig_path(savefig, value, num_values):
    ext_set = {'.png', '.pdf', '.svg', '.jpg', '.jpeg'}
    root, ext = osp.splitext(savefig)
    if num_values == 1 and ext.lower() in ext_set:
        parent = osp.dirname(savefig)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return savefig
    if not ext or ext.lower() not in ext_set:
        os.makedirs(savefig, exist_ok=True)
        return osp.join(savefig, f'plot_{value}.png')
    parent = osp.dirname(savefig)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return f'{root}_{value}{ext}'


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean',
               savefig=None):
    data, logdirs = _collect_logdirs_and_datasets(all_logdirs, legend, select, exclude)
    if not data:
        raise ValueError(f"No experiment progress.txt files found matching: {all_logdirs}")
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    figs = []
    for value in values:
        fig = plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition,
                  smooth=smooth, font_scale=font_scale, estimator=estimator)
        figs.append((fig, value))

    if savefig is not None:
        for fig, value in figs:
            out_path = _resolve_savefig_path(savefig, value, len(figs))
            fig.savefig(out_path, bbox_inches='tight')
            print('Saved plot to %s' % out_path)
    elif _is_interactive_backend():
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument(
        '--savefig', '-o', type=str, default=None,
        help='Save figure(s) to file or directory instead of opening an interactive window.',
    )
    args = parser.parse_args()
    """

    Args: 

        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

        savefig (string): Optional output file path or directory to save figures
            to instead of opening an interactive GUI window.

    """
    savefig = args.savefig
    if savefig is None and not _is_interactive_backend():
        if len(args.logdir) == 1 and osp.isdir(args.logdir[0]):
            savefig = args.logdir[0].rstrip(os.sep) or '.'
        else:
            savefig = '.'
        print(
            f'No interactive display detected; saving plot(s) to {savefig} '
            '(pass --savefig/-o <path> to choose a custom location).'
        )

    try:
        make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
                   smooth=args.smooth, select=args.select, exclude=args.exclude,
                   estimator=args.est, savefig=savefig)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
