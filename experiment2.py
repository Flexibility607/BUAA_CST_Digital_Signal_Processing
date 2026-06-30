from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


FS = 10000
N = 1600
RP = 0.1
RS = 60
OUT_DIR = Path(__file__).resolve().parent / "figures" / "experiment2"

COMPONENTS = [
    {"name": "y1", "fc": 1000, "fm": 100, "label": "fc=1000 Hz, fm=100 Hz"},
    {"name": "y2", "fc": 500, "fm": 50, "label": "fc=500 Hz, fm=50 Hz"},
    {"name": "y3", "fc": 250, "fm": 25, "label": "fc=250 Hz, fm=25 Hz"},
]

FILTER_SPECS = {
    "lowpass_y3": {
        "title": "Low-pass filter for y3",
        "btype": "lowpass",
        "wp": 300,
        "ws": 400,
        "passbands": [(0, 300)],
        "stopbands": [(400, FS / 2)],
        "target": "y3",
    },
    "bandpass_y2": {
        "title": "Band-pass filter for y2",
        "btype": "bandpass",
        "wp": [400, 600],
        "ws": [350, 700],
        "passbands": [(400, 600)],
        "stopbands": [(0, 350), (700, FS / 2)],
        "target": "y2",
    },
    "highpass_y1": {
        "title": "High-pass filter for y1",
        "btype": "highpass",
        "wp": 800,
        "ws": 700,
        "passbands": [(800, FS / 2)],
        "stopbands": [(0, 700)],
        "target": "y1",
    },
}


def set_common_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def make_time_axis(n_samples=N):
    return np.arange(n_samples) / FS


def scam_component(t, fc, fm):
    return np.cos(2 * np.pi * fm * t) * np.cos(2 * np.pi * fc * t)


def am_component(t, fc, fm, a0=1.0, am=1.0):
    return (a0 + am * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)


def build_signal(n_samples=N, am=False):
    t = make_time_axis(n_samples)
    components = {}
    for item in COMPONENTS:
        if am:
            components[item["name"]] = am_component(t, item["fc"], item["fm"])
        else:
            components[item["name"]] = scam_component(t, item["fc"], item["fm"])
    mixed = sum(components.values())
    return t, mixed, components


def expected_lines():
    lines = []
    for item in COMPONENTS:
        lines.extend([item["fc"] - item["fm"], item["fc"] + item["fm"]])
    return sorted(lines)


def single_sided_spectrum(x, fs=FS, nfft=None, normalize=True):
    if nfft is None:
        nfft = len(x)
    freq = np.fft.rfftfreq(nfft, d=1 / fs)
    spectrum = np.fft.rfft(x, nfft)
    mag = np.abs(spectrum)
    if normalize and mag.max() > 0:
        mag = mag / mag.max()
    return freq, mag


def design_filters():
    designed = {}
    for key, spec in FILTER_SPECS.items():
        order, wn = signal.ellipord(spec["wp"], spec["ws"], RP, RS, fs=FS)
        sos = signal.ellip(order, RP, RS, wn, btype=spec["btype"], output="sos", fs=FS)
        designed[key] = {**spec, "order": order, "wn": wn, "sos": sos}
    return designed


def response_loss_db(sos, wor_n=32768):
    freq, response = signal.sosfreqz(sos, worN=wor_n, fs=FS)
    magnitude = np.maximum(np.abs(response), 1e-14)
    loss = -20 * np.log10(magnitude)
    return freq, loss


def band_mask(freq, bands):
    mask = np.zeros_like(freq, dtype=bool)
    for low, high in bands:
        mask |= (freq >= low) & (freq <= high)
    return mask


def summarize_filter(designed):
    lines = []
    lines.append(f"Elliptic filter specification: rp={RP} dB, rs={RS} dB")
    for key, spec in designed.items():
        freq, loss = response_loss_db(spec["sos"])
        pass_mask = band_mask(freq, spec["passbands"])
        stop_mask = band_mask(freq, spec["stopbands"])
        pass_loss = float(loss[pass_mask].max())
        stop_atten = float(loss[stop_mask].min())
        spec["pass_loss_db"] = pass_loss
        spec["stop_atten_db"] = stop_atten
        lines.append(
            f"{key}: order={spec['order']}, passband max loss={pass_loss:.4f} dB, "
            f"stopband min attenuation={stop_atten:.4f} dB"
        )
    return lines


def plot_mixed_signal():
    t, st, _ = build_signal(N, am=False)
    freq, mag = single_sided_spectrum(st)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), constrained_layout=True)
    time_mask = t <= 0.02
    axes[0].plot(t[time_mask] * 1000, st[time_mask], color="C0", linewidth=1.2)
    axes[0].set_title("Mixed SCAM signal")
    axes[0].set_xlabel("time / ms")
    axes[0].set_ylabel("s(t)")

    freq_mask = freq <= 1200
    markerline, stemlines, baseline = axes[1].stem(
        freq[freq_mask],
        mag[freq_mask],
        basefmt=" ",
        linefmt="C1-",
        markerfmt="C1.",
    )
    plt.setp(stemlines, linewidth=0.8)
    plt.setp(markerline, markersize=4)
    for line in expected_lines():
        axes[1].axvline(line, color="0.55", linewidth=0.8, linestyle="--")
    axes[1].set_title("N-point FFT magnitude of mixed signal")
    axes[1].set_xlabel("frequency / Hz")
    axes[1].set_ylabel("normalized magnitude")
    axes[1].set_xlim(0, 1200)
    axes[1].set_ylim(0, 1.08)

    path = OUT_DIR / "mixed_signal.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_filters_loss(designed):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), constrained_layout=True)
    colors = ["C0", "C1", "C2"]
    for ax, (color, (key, spec)) in zip(axes, zip(colors, designed.items())):
        freq, loss = response_loss_db(spec["sos"])
        ax.plot(freq, np.clip(loss, 0, 100), color=color, linewidth=1.35)
        ax.axhline(RP, color="0.35", linestyle="--", linewidth=0.8, label="0.1 dB")
        ax.axhline(RS, color="0.35", linestyle=":", linewidth=0.8, label="60 dB")
        for low, high in spec["passbands"]:
            ax.axvspan(low, high, color="C0", alpha=0.08)
        for low, high in spec["stopbands"]:
            ax.axvspan(low, min(high, 1200), color="C3", alpha=0.05)
        ax.set_title(f"{spec['title']} (order {spec['order']})")
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("loss / dB")
        ax.set_xlim(0, 1200)
        ax.set_ylim(-1, 92)
        ax.legend(loc="upper right")

    path = OUT_DIR / "filters_loss.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def filter_outputs(st, designed):
    outputs = {}
    for key, spec in designed.items():
        outputs[spec["target"]] = signal.sosfilt(spec["sos"], st)
    return outputs


def plot_separated_signals(outputs):
    t = make_time_axis(N)
    order = [
        ("y3", "Low-pass output y3, fc=250 Hz"),
        ("y2", "Band-pass output y2, fc=500 Hz"),
        ("y1", "High-pass output y1, fc=1000 Hz"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8.6), constrained_layout=True)
    time_mask = (t >= 0.02) & (t <= 0.08)

    for row, (name, title) in enumerate(order):
        y = outputs[name]
        axes[row, 0].plot(t[time_mask] * 1000, y[time_mask], linewidth=1.0)
        axes[row, 0].set_title(title)
        axes[row, 0].set_xlabel("time / ms")
        axes[row, 0].set_ylabel(f"{name}(n)")

        freq, mag = single_sided_spectrum(y)
        freq_mask = freq <= 1200
        axes[row, 1].plot(freq[freq_mask], mag[freq_mask], linewidth=1.25, color=f"C{row + 1}")
        axes[row, 1].set_title(f"Spectrum of {name}(n)")
        axes[row, 1].set_xlabel("frequency / Hz")
        axes[row, 1].set_ylabel("normalized magnitude")
        axes[row, 1].set_xlim(0, 1200)
        axes[row, 1].set_ylim(0, 1.08)

    path = OUT_DIR / "separated_signals.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fft_length_comparison():
    lengths = [1600, 1800, 2000]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), constrained_layout=True)
    checks = []
    for ax, n_samples in zip(axes, lengths):
        _, st, _ = build_signal(n_samples, am=False)
        freq, mag = single_sided_spectrum(st, nfft=n_samples)
        freq_mask = freq <= 1200
        ax.plot(freq[freq_mask], mag[freq_mask], linewidth=1.05)
        for line in expected_lines():
            ax.axvline(line, color="0.55", linewidth=0.8, linestyle="--")
        bin_spacing = FS / n_samples
        integer_bins = [abs(line * n_samples / FS - round(line * n_samples / FS)) < 1e-9 for line in expected_lines()]
        all_ideal = all(integer_bins)
        non_integer = [line for line, ok in zip(expected_lines(), integer_bins) if not ok]
        checks.append((n_samples, bin_spacing, all_ideal, non_integer))
        ax.set_title(
            f"N={n_samples}, bin spacing={bin_spacing:.4f} Hz, "
            f"all six lines on DFT bins: {all_ideal}"
        )
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("normalized magnitude")
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 1.08)

    path = OUT_DIR / "fft_length_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, checks


def plot_am_vs_scam():
    t, scam, _ = build_signal(N, am=False)
    _, am_sig, _ = build_signal(N, am=True)
    scam_freq, scam_mag = single_sided_spectrum(scam)
    am_freq, am_mag = single_sided_spectrum(am_sig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0), constrained_layout=True)
    time_mask = t <= 0.03
    axes[0, 0].plot(t[time_mask] * 1000, scam[time_mask], linewidth=1.0)
    axes[0, 0].set_title("SCAM mixed signal")
    axes[0, 0].set_xlabel("time / ms")
    axes[0, 0].set_ylabel("amplitude")

    axes[0, 1].plot(t[time_mask] * 1000, am_sig[time_mask], color="C1", linewidth=1.0)
    axes[0, 1].set_title("AM mixed signal, A0=Am=1")
    axes[0, 1].set_xlabel("time / ms")
    axes[0, 1].set_ylabel("amplitude")

    freq_mask = scam_freq <= 1200
    axes[1, 0].stem(
        scam_freq[freq_mask],
        scam_mag[freq_mask],
        basefmt=" ",
        linefmt="C0-",
        markerfmt="C0.",
    )
    axes[1, 0].set_title("SCAM spectrum")
    axes[1, 0].set_xlabel("frequency / Hz")
    axes[1, 0].set_ylabel("normalized magnitude")
    axes[1, 0].set_xlim(0, 1200)
    axes[1, 0].set_ylim(0, 1.08)

    freq_mask = am_freq <= 1200
    axes[1, 1].stem(
        am_freq[freq_mask],
        am_mag[freq_mask],
        basefmt=" ",
        linefmt="C1-",
        markerfmt="C1.",
    )
    for carrier in [250, 500, 1000]:
        axes[1, 1].axvline(carrier, color="0.55", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("AM spectrum with carrier components")
    axes[1, 1].set_xlabel("frequency / Hz")
    axes[1, 1].set_ylabel("normalized magnitude")
    axes[1, 1].set_xlim(0, 1200)
    axes[1, 1].set_ylim(0, 1.08)

    path = OUT_DIR / "am_vs_scam.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def spectral_line_summary(st):
    freq, mag = single_sided_spectrum(st, normalize=True)
    lines = ["Expected positive-frequency SCAM lines:"]
    for line in expected_lines():
        idx = int(np.argmin(np.abs(freq - line)))
        lines.append(f"{line:7.1f} Hz -> nearest DFT bin {freq[idx]:7.2f} Hz, normalized magnitude {mag[idx]:.4f}")
    return lines


def write_summary(designed, fft_checks, figure_paths, st):
    summary = []
    summary.extend(summarize_filter(designed))
    summary.append("")
    summary.extend(spectral_line_summary(st))
    summary.append("")
    summary.append("FFT length ideal-line check:")
    for n_samples, bin_spacing, all_ideal, non_integer in fft_checks:
        if non_integer:
            missing = ", ".join(f"{item:g} Hz" for item in non_integer)
        else:
            missing = "none"
        summary.append(
            f"N={n_samples}: bin spacing={bin_spacing:.6f} Hz, all ideal={all_ideal}, "
            f"non-integer-bin lines={missing}"
        )
    summary.append("")
    summary.append("Generated figures:")
    for path in figure_paths:
        summary.append(f"- {path.as_posix()}")

    summary_path = OUT_DIR / "verification_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return summary_path, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_common_style()

    t, st, _ = build_signal(N, am=False)
    designed = design_filters()
    outputs = filter_outputs(st, designed)

    figure_paths = [
        plot_mixed_signal(),
        plot_filters_loss(designed),
        plot_separated_signals(outputs),
    ]
    fft_path, fft_checks = plot_fft_length_comparison()
    figure_paths.append(fft_path)
    figure_paths.append(plot_am_vs_scam())

    summary_path, summary = write_summary(designed, fft_checks, figure_paths, st)

    print("\n".join(summary))
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
