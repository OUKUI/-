import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def render_profile(result, view="full", scale_mag=1.0):
    """Generate roundness profile figure, return numpy array (H, W, 3).
    scale_mag: multiplier for deviation radial range (1.0 = auto)."""
    r = result
    angles = np.arctan2(r.points[:, 1] - r.cy, r.points[:, 0] - r.cx)
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    errors_sorted = r.errors[sort_idx]
    angles_loop = np.append(angles_sorted, angles_sorted[0] + 2 * np.pi)
    errors_loop = np.append(errors_sorted, errors_sorted[0])
    max_err = max(abs(r.peak_error), abs(r.valley_error), 1e-6)
    sm = scale_mag

    fig = plt.figure(figsize=(7, 6) if view != "full" else (10, 7), facecolor='#1a1a2e')

    if view == "full":
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1], hspace=0.25, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0], polar=True)
        ax1.set_facecolor('#12121e')
        profile_r = r.radius + errors_sorted
        ax1.plot(angles_loop, np.full_like(angles_loop, r.radius), '--', color='#555577', lw=1, alpha=0.6)
        ax1.plot(angles_sorted, profile_r, 'o-', color='#60a5fa', lw=1.5, ms=3)
        ax1.fill(angles_loop, r.radius + errors_loop, alpha=0.12, color='#60a5fa')
        ax1.plot(angles_loop, np.full_like(angles_loop, r.radius + r.peak_error), ':', color='#f87171', lw=0.6)
        ax1.plot(angles_loop, np.full_like(angles_loop, r.radius + r.valley_error), ':', color='#4ade80', lw=0.6)
        ax1.set_title(f'Profile  [R={r.radius:.2f}]', color='#e8e8f0', fontsize=10, pad=12)
        ax1.tick_params(colors='#9898b8', labelsize=7)
        ax1.grid(True, alpha=0.15, color='#555577')

        rlim = max_err * 1.4 * sm
        offset = rlim
        dev_shift = errors_sorted + offset
        ax2 = fig.add_subplot(gs[0, 1], polar=True)
        ax2.set_facecolor('#12121e')
        ax2.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                         where=(errors_sorted >= 0), color='#f87171')
        ax2.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                         where=(errors_sorted < 0), color='#4ade80')
        ax2.plot(angles_sorted, dev_shift, '-', color='#93c5fd', lw=1.5)
        ax2.plot(angles_sorted, dev_shift, '.', color='#e0e7ff', ms=3)
        ax2.plot(angles_loop, np.full_like(angles_loop, offset), '-', color='#555577', lw=1, alpha=0.5)
        ax2.plot(angles_loop, np.full_like(angles_loop, r.peak_error + offset), ':', color='#f87171', lw=0.6)
        ax2.plot(angles_loop, np.full_like(angles_loop, r.valley_error + offset), ':', color='#4ade80', lw=0.6)
        ax2.set_ylim(0, 2 * offset)
        tick_vals = np.linspace(0, 2 * offset, 5)
        ax2.set_yticks(tick_vals)
        ax2.set_yticklabels([f'{v - offset:.3f}' for v in tick_vals], color='#9898b8', fontsize=6)
        ax2.set_title(f'Deviation (polar)  [\u00b1{offset:.4f}]', color='#e8e8f0', fontsize=10, pad=12)
        ax2.tick_params(colors='#9898b8', labelsize=7)
        ax2.grid(True, alpha=0.15, color='#555577')

        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor('#12121e')
        ax3.axhline(0, color='#555577', lw=1, ls='--', alpha=0.5)
        ax3.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3, where=(errors_sorted >= 0), color='#f87171')
        ax3.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3, where=(errors_sorted < 0), color='#4ade80')
        ax3.plot(angles_sorted, errors_sorted, '-', color='#60a5fa', lw=1.5)
        ax3.plot(angles_sorted, errors_sorted, '.', color='#93c5fd', ms=4)
        ax3.axhline(r.peak_error, color='#f87171', lw=0.6, ls=':')
        ax3.axhline(r.valley_error, color='#4ade80', lw=0.6, ls=':')
        ax3.set_xlim(-0.1, 2 * np.pi + 0.1)
        ax3.set_title('Deviation (linear)', color='#e8e8f0', fontsize=11)
        ax3.set_xlabel('Angle (rad)', color='#9898b8', fontsize=8)
        ax3.set_ylabel('Error', color='#9898b8', fontsize=8)
        ax3.tick_params(colors='#9898b8', labelsize=7)
        ax3.grid(True, alpha=0.15, color='#555577')

    elif view == "profile":
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor('#12121e')
        ax.plot(angles_loop, np.full_like(angles_loop, r.radius), '--', color='#555577', lw=1, alpha=0.6)
        ax.plot(angles_sorted, r.radius + errors_sorted, 'o-', color='#60a5fa', lw=2, ms=4)
        ax.fill(angles_loop, r.radius + errors_loop, alpha=0.15, color='#60a5fa')
        ax.plot(angles_loop, np.full_like(angles_loop, r.radius + r.peak_error), ':', color='#f87171', lw=0.8)
        ax.plot(angles_loop, np.full_like(angles_loop, r.radius + r.valley_error), ':', color='#4ade80', lw=0.8)
        ax.set_title(f'Roundness Profile  [R={r.radius:.2f}]', color='#e8e8f0', fontsize=12, pad=16)
        ax.tick_params(colors='#9898b8', labelsize=8)
        ax.grid(True, alpha=0.15, color='#555577')

    elif view == "deviation_polar":
        rlim = max_err * 1.4 * sm
        offset = rlim
        dev_shift = errors_sorted + offset
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor('#12121e')
        ax.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                        where=(errors_sorted >= 0), color='#f87171')
        ax.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                        where=(errors_sorted < 0), color='#4ade80')
        ax.plot(angles_sorted, dev_shift, '-', color='#93c5fd', lw=2)
        ax.plot(angles_sorted, dev_shift, '.', color='#e0e7ff', ms=4)
        ax.plot(angles_loop, np.full_like(angles_loop, offset), '-', color='#555577', lw=1, alpha=0.5)
        ax.plot(angles_loop, np.full_like(angles_loop, r.peak_error + offset), ':', color='#f87171', lw=0.8)
        ax.plot(angles_loop, np.full_like(angles_loop, r.valley_error + offset), ':', color='#4ade80', lw=0.8)
        ax.set_ylim(0, 2 * offset)
        ax.set_yticks(np.linspace(0, 2 * offset, 5))
        ax.set_yticklabels([f'{v - offset:.3f}' for v in np.linspace(0, 2 * offset, 5)], color='#9898b8', fontsize=7)
        ax.set_title(f'Deviation (polar)  [\u00b1{offset:.4f}]', color='#e8e8f0', fontsize=12, pad=16)
        ax.tick_params(colors='#9898b8', labelsize=8)
        ax.grid(True, alpha=0.2, color='#555577')

    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor('#12121e')
        ax.axhline(0, color='#555577', lw=1, ls='--', alpha=0.5)
        ax.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3,
                        where=(errors_sorted >= 0), color='#f87171')
        ax.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3,
                        where=(errors_sorted < 0), color='#4ade80')
        ax.plot(angles_sorted, errors_sorted, '-', color='#60a5fa', lw=2)
        ax.plot(angles_sorted, errors_sorted, '.', color='#93c5fd', ms=4)
        ax.axhline(r.peak_error, color='#f87171', lw=0.6, ls=':')
        ax.axhline(r.valley_error, color='#4ade80', lw=0.6, ls=':')
        ax.set_xlim(-0.1, 2 * np.pi + 0.1)
        ax.set_title('Deviation (linear)', color='#e8e8f0', fontsize=12)
        ax.set_xlabel('Angle (rad)', color='#9898b8', fontsize=9)
        ax.set_ylabel('Error', color='#9898b8', fontsize=9)
        ax.tick_params(colors='#9898b8', labelsize=8)
        ax.grid(True, alpha=0.15, color='#555577')

    summary = (
        f"C: ({r.cx:.3f}, {r.cy:.3f})  R: {r.radius:.3f}  "
        f"Peak: +{r.peak_error:.4f}  Valley: {r.valley_error:.4f}  "
        f"Roundness: {r.roundness:.4f}  RMSE: {r.rmse:.6f}"
    )
    fig.text(0.5, 0.005, summary, ha='center', va='bottom', color='#9898b8', fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    import cv2
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return img
