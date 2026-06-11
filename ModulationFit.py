import numpy as np
from scipy.optimize import curve_fit
import argparse
import matplotlib.pyplot as plt

# Fit for the modulation
def polarfit(x, A, phi0, N):
    return N / (2 * np.pi) * (1 - A * np.cos(2 * (x - phi0)))

'''
MAIN FUNCTION
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarized", required=True, help="Path to polarized phi values .txt file")
    parser.add_argument("--unpolarized", required=True, help="Path to unpolarized phi values .txt file")
    args = parser.parse_args()

    # Getting the polarized and unpolarized phi values
    phi_polarized = np.loadtxt(args.polarized)
    phi_unpolarized = np.loadtxt(args.unpolarized)

    # Histogramming
    bins = np.linspace(-np.pi, np.pi, 17)
    x = 0.5 * (bins[:-1] + bins[1:])

    h_pol, _ = np.histogram(phi_polarized, bins=bins)
    h_unpol, _ = np.histogram(phi_unpolarized, bins=bins)

    # Phi distributions
    plt.figure(figsize=(8, 4))
    plt.hist(phi_polarized, bins=bins, alpha=0.5, label="Polarized")
    plt.hist(phi_unpolarized, bins=bins, alpha=0.5, label="Unpolarized")
    plt.xlabel(r"Azimuthal angle $\varphi$ [rad]")
    plt.ylabel("Counts")
    plt.title("Histogram of Polarized and Unpolarized Azimuthal Angles")
    plt.legend()
    plt.tight_layout()
    plt.show()

    valid = (h_pol > 0) & (h_unpol > 0)
    x = x[valid]
    h_pol = h_pol[valid]
    h_unpol = h_unpol[valid]

    ratio = h_pol / h_unpol
    error = ratio * np.sqrt(1/h_pol + 1/h_unpol)

    # Fitting the modulation curve to the ratio
    plt.figure(figsize=(8, 4))
    plt.errorbar(x, ratio, yerr=error, fmt='o', label='Polarized / Unpolarized')

    try:
        popt, pcov = curve_fit(polarfit, x, ratio, sigma=error, absolute_sigma=True)
        A_fit, phi0_fit, N_fit = popt
        dA, dphi0, dN = np.sqrt(np.diag(pcov))

        print(f"A     = {A_fit:.3f} ± {dA:.3f}")
        print(f"phi_0 = {phi0_fit:.3f} ± {dphi0:.3f} rad")
        print(f"N     = {N_fit:.1f} ± {dN:.1f}")

        # Compute chi squared and ndof values
        residuals = ratio - polarfit(x, *popt)
        chi2 = np.sum((residuals / error)**2)
        ndof = len(x) - len(popt)

        print(f"Chi-squared = {chi2:.2f}")
        print(f"ndof        = {ndof}")
        print(f"Chi2/ndof   = {chi2/ndof:.2f}")

        xx = np.linspace(-np.pi, np.pi, 500)
        plt.plot(xx, polarfit(xx, *popt), 'r--', label="Polarization Fit")

    except RuntimeError:
        print('Fit failed.')

    plt.xlabel(r"Azimuthal angle $\varphi$ [rad]")
    plt.ylabel("Polarized / Unpolarized")
    plt.title("Modulation Fit")
    plt.grid(linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()