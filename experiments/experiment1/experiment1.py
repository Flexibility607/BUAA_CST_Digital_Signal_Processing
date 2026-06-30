import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_common_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def analog_signal_samples(fs, tp=0.05):
    a = 444.128
    alpha = 50 * np.sqrt(2) * np.pi
    omega0 = 50 * np.sqrt(2) * np.pi
    n = np.arange(int(round(tp * fs)))
    t = n / fs
    x = a * np.exp(-alpha * t) * np.sin(omega0 * t)
    return n, t, x


def plot_time_domain_sampling():
    fft_len = 64
    sample_rates = [1000, 300, 200]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)

    for row, fs in enumerate(sample_rates):
        n, t, x = analog_signal_samples(fs)
        padded = np.zeros(fft_len)
        padded[: len(x)] = x
        spectrum = np.fft.fftshift(np.fft.fft(padded, fft_len))
        freq = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1 / fs))
        mag = np.abs(spectrum)
        if mag.max() > 0:
            mag = mag / mag.max()

        ax_time = axes[row, 0]
        ax_freq = axes[row, 1]

        ax_time.stem(t * 1000, x, basefmt=" ", linefmt="C0-", markerfmt="C0o")
        ax_time.set_title(f"Fs = {fs} Hz, N = {len(x)}")
        ax_time.set_xlabel("time / ms")
        ax_time.set_ylabel("x[n]")

        ax_freq.plot(freq, mag, color="C1", linewidth=1.6)
        ax_freq.set_title(f"64-point FFT magnitude, Fs = {fs} Hz")
        ax_freq.set_xlabel("frequency / Hz")
        ax_freq.set_ylabel("normalized |X[k]|")
        ax_freq.set_xlim(-fs / 2, fs / 2)
        ax_freq.set_ylim(0, 1.08)

    fig.suptitle("Time-domain sampling verification", fontsize=13)
    fig.savefig(OUT_DIR / "time_domain_sampling.png", bbox_inches="tight")
    plt.close(fig)


def build_frequency_domain_sequence():
    n = np.arange(27)
    x = np.zeros_like(n, dtype=float)
    x[(0 <= n) & (n <= 13)] = n[(0 <= n) & (n <= 13)] + 1
    x[(14 <= n) & (n <= 26)] = 27 - n[(14 <= n) & (n <= 26)]
    return n, x


def periodize_to_length(x, period):
    folded = np.zeros(period, dtype=float)
    for i, value in enumerate(x):
        folded[i % period] += value
    return folded


def plot_frequency_domain_sampling():
    n, x = build_frequency_domain_sequence()

    x32 = np.zeros(32)
    x32[: len(x)] = x
    x16_alias = periodize_to_length(x, 16)

    x32_spec = np.fft.fft(x32, 32)
    x16_spec = x32_spec[::2]
    x32_ifft = np.fft.ifft(x32_spec).real
    x16_ifft = np.fft.ifft(x16_spec).real

    dense_n = np.arange(len(x))
    omega = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    dense_spec = np.array(
        [np.sum(x * np.exp(-1j * w * dense_n)) for w in omega]
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    axes[0, 0].plot(omega / np.pi, np.abs(dense_spec), color="C0", linewidth=1.6)
    axes[0, 0].set_title("DTFT magnitude")
    axes[0, 0].set_xlabel(r"$\omega/\pi$")
    axes[0, 0].set_ylabel(r"$|X(e^{j\omega})|$")
    axes[0, 0].set_xlim(0, 2)

    k32 = np.arange(32)
    axes[0, 1].stem(
        2 * k32 / 32,
        np.abs(x32_spec),
        basefmt=" ",
        linefmt="C1-",
        markerfmt="C1o",
    )
    axes[0, 1].set_title("32 samples in frequency")
    axes[0, 1].set_xlabel(r"$\omega/\pi$")
    axes[0, 1].set_ylabel(r"$|X_{32}[k]|$")
    axes[0, 1].set_xlim(0, 2)

    k16 = np.arange(16)
    axes[0, 2].stem(
        2 * k16 / 16,
        np.abs(x16_spec),
        basefmt=" ",
        linefmt="C2-",
        markerfmt="C2o",
    )
    axes[0, 2].set_title("16 samples in frequency")
    axes[0, 2].set_xlabel(r"$\omega/\pi$")
    axes[0, 2].set_ylabel(r"$|X_{16}[k]|$")
    axes[0, 2].set_xlim(0, 2)

    axes[1, 0].stem(n, x, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    axes[1, 0].set_title("Original x[n], M = 27")
    axes[1, 0].set_xlabel("n")
    axes[1, 0].set_ylabel("x[n]")

    axes[1, 1].stem(
        np.arange(32),
        x32_ifft,
        basefmt=" ",
        linefmt="C1-",
        markerfmt="C1o",
    )
    axes[1, 1].set_title("IFFT of 32 frequency samples")
    axes[1, 1].set_xlabel("n")
    axes[1, 1].set_ylabel(r"$x_{32}[n]$")

    axes[1, 2].stem(
        np.arange(16),
        x16_ifft,
        basefmt=" ",
        linefmt="C2-",
        markerfmt="C2o",
    )
    axes[1, 2].plot(
        np.arange(16),
        x16_alias,
        "k--",
        linewidth=1,
        label="periodic alias sum",
    )
    axes[1, 2].set_title("IFFT of 16 frequency samples")
    axes[1, 2].set_xlabel("n")
    axes[1, 2].set_ylabel(r"$x_{16}[n]$")
    axes[1, 2].legend(frameon=False)

    fig.suptitle("Frequency-domain sampling verification", fontsize=13)
    fig.savefig(OUT_DIR / "frequency_domain_sampling.png", bbox_inches="tight")
    plt.close(fig)

    return x16_alias, x16_ifft


def main():
    set_common_style()
    plot_time_domain_sampling()
    x16_alias, x16_ifft = plot_frequency_domain_sampling()
    print("Generated figures:")
    print(f"- {OUT_DIR / 'time_domain_sampling.png'}")
    print(f"- {OUT_DIR / 'frequency_domain_sampling.png'}")
    print("x16 periodic alias sum:")
    print(np.round(x16_alias, 6))
    print("x16 from IFFT:")
    print(np.round(x16_ifft, 6))


if __name__ == "__main__":
    main()
