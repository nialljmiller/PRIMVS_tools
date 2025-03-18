import numpy as np
import matplotlib.pyplot as plt
import Virac

# Assuming 'Virac.run_sourceid' is a function that fetches the lightcurve data for a given source ID.

def phaser(time, period):
    phase = np.mod(time, period) / period
    return phase

def plot_light_curves(names,periods,outputfp):
    n = len(names)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    print('###############',names)
    for i, name in enumerate(names):
        period = periods[i]
        print(name,period)
        lightcurve = Virac.run_sourceid(int(name))
        filters = (lightcurve['filter'].astype(str) == 'Ks')
        mag_gt_0 = (lightcurve['hfad_mag'].astype(float) > 0)
        emag_gt_0 = (lightcurve['hfad_emag'].astype(float) > 0)
        emag_lt = (lightcurve['hfad_emag'].astype(float) < 0.1)
        ast_res_chisq_lt_20 = (lightcurve['ast_res_chisq'].astype(float) < 20)
        chi_lt_10 = (lightcurve['chi'].astype(float) < 10)
        filtered_indices = np.where(filters & mag_gt_0 & emag_gt_0 & ast_res_chisq_lt_20 & chi_lt_10 & emag_lt)[0]
        lightcurve = lightcurve[filtered_indices]
        #print(lightcurve)
        mag, magerr, time, chi, astchi = lightcurve['hfad_mag'], lightcurve['hfad_emag'], lightcurve['mjdobs'], lightcurve['chi'], lightcurve['ast_res_chisq']

        phase = phaser(time, period)

        norm = plt.Normalize(min(time), max(time))
        cmap = plt.cm.viridis  # You can choose any colormap you like

        row, col = divmod(i, ncols)
        ax = axes[row, col] if n > 1 else axes

        # Plot error bars in neutral color without markers
        #ax.errorbar(phase, mag, yerr=magerr, fmt='x', ecolor='k', capsize=0)
        #ax.errorbar(phase+1, mag, yerr=magerr, fmt='x', ecolor='k', capsize=0)

        # Plot points colored by mjd using scatter
        sc = ax.scatter(phase, mag, c=time, s=1, cmap=cmap, norm=norm)
        ax.scatter(phase+1, mag, c=time, s=1, cmap=cmap, norm=norm)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()  # Because magnitude decreases as brightness increases
        ax.set_ylim(np.percentile(mag, [1,99]))

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(outputfp + 'phase.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    print('###############',names)
    for i, name in enumerate(names):
        period = periods[i]
        print(name,period)
        lightcurve = Virac.run_sourceid(int(name))
        filters = (lightcurve['filter'].astype(str) == 'Ks')
        mag_gt_0 = (lightcurve['hfad_mag'].astype(float) > 0)
        emag_gt_0 = (lightcurve['hfad_emag'].astype(float) > 0)
        emag_lt = (lightcurve['hfad_emag'].astype(float) < 0.1)
        ast_res_chisq_lt_20 = (lightcurve['ast_res_chisq'].astype(float) < 20)
        chi_lt_10 = (lightcurve['chi'].astype(float) < 10)
        filtered_indices = np.where(filters & mag_gt_0 & emag_gt_0 & ast_res_chisq_lt_20 & chi_lt_10 & emag_lt)[0]
        lightcurve = lightcurve[filtered_indices]
        #print(lightcurve)
        mag, magerr, time, chi, astchi = lightcurve['hfad_mag'], lightcurve['hfad_emag'], lightcurve['mjdobs'], lightcurve['chi'], lightcurve['ast_res_chisq']
        row, col = divmod(i, ncols)
        ax = axes[row, col] if n > 1 else axes
        ax.scatter(time, mag, s=1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()  # Because magnitude decreases as brightness increases
        ax.set_ylim(np.percentile(mag, [1,99]))

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(outputfp + 'mjd.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


    n = len(names)
    ncols = int(np.ceil(np.sqrt(n)))*2  # Double the columns for pairs
    nrows = int(np.ceil(n / (ncols / 2)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16,9), squeeze=False)
    fig.subplots_adjust(wspace=0)  # This removes the horizontal gap between subplots
    for i, name in enumerate(names):
        period = periods[i]
        print(name,period)
        lightcurve = Virac.run_sourceid(int(name))
        filters = (lightcurve['filter'].astype(str) == 'Ks')
        mag_gt_0 = (lightcurve['hfad_mag'].astype(float) > 0)
        emag_gt_0 = (lightcurve['hfad_emag'].astype(float) > 0)
        emag_lt = (lightcurve['hfad_emag'].astype(float) < 0.1)
        ast_res_chisq_lt_20 = (lightcurve['ast_res_chisq'].astype(float) < 20)
        chi_lt_10 = (lightcurve['chi'].astype(float) < 10)
        filtered_indices = np.where(filters & mag_gt_0 & emag_gt_0 & ast_res_chisq_lt_20 & chi_lt_10 & emag_lt)[0]
        lightcurve = lightcurve[filtered_indices]
        #print(lightcurve)
        mag, magerr, time, chi, astchi = lightcurve['hfad_mag'], lightcurve['hfad_emag'], lightcurve['mjdobs'], lightcurve['chi'], lightcurve['ast_res_chisq']

        phase = phaser(time, period)

        norm = plt.Normalize(min(time), max(time))
        cmap = plt.cm.viridis

        # Calculate positions for phase/mag and time/mag plots
        row = i // (ncols // 2)
        col_phase = (i * 2) % ncols
        col_time = col_phase + 1

        # Phase vs Mag plot
        ax_phase = axes[row, col_phase]
        ax_phase.scatter(phase, mag, c=time, s=1, cmap=cmap, norm=norm)
        ax_phase.scatter(phase + 1, mag, c=time, s=1, cmap=cmap, norm=norm)
        ax_phase.set_xticklabels([])
        ax_phase.set_yticklabels([])
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])
        ax_phase.invert_yaxis()
        ax_phase.set_ylim(np.percentile(mag, [1, 99]))

        # Time vs Mag plot
        ax_time = axes[row, col_time]
        ax_time.scatter(time, mag, c=time, s=1, cmap=cmap, norm=norm)
        ax_time.set_xticklabels([])
        ax_time.set_yticklabels([])
        ax_time.set_xticks([])
        ax_time.set_yticks([])
        ax_time.invert_yaxis()
        ax_time.set_ylim(np.percentile(mag, [1, 99]))

    # Hide unused subplots
    for j in range(i * 2 + 2, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    plt.savefig(outputfp + '.jpg', dpi=300, bbox_inches='tight')
    plt.clf()




for filename in [1,2,3,4,5,6,7,8,9]:
    # Example usage
    names, periods = np.genfromtxt('/beegfs/car/njm/PRIMVS/autoencoder/groups/'+str(filename)+'.csv', delimiter=',', skip_header=1, unpack = True, usecols = [1,107])

    outputfp = '/beegfs/car/njm/PRIMVS/autoencoder/groups/' + str(filename)

    plot_light_curves(names,periods,outputfp)

