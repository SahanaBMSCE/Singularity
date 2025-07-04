import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lightkurve as lk
import warnings
warnings.filterwarnings('ignore')

class Kepler10Analysis:
    def __init__(self):
        self.target_name = "Kepler-10"
        self.light_curve = None
        self.period = 0.837495  
        self.t0 = 131.51
    def download_and_process(self):
        print(f"Downloading {self.target_name} data from Kepler mission...")
        search_result = lk.search_lightcurve(self.target_name, mission="Kepler")
        if len(search_result) == 0:
            print(f"No data found for {self.target_name}")
            return None 
        print(f"Found {len(search_result)} quarters of data")
        lc_collection = search_result[:6].download_all()
        self.light_curve = lc_collection.stitch()
        self.light_curve = self.light_curve.remove_outliers(sigma=3)
        self.light_curve = self.light_curve.normalize()
        print(f"Processed {len(self.light_curve.time)} data points")
        time_span_days = (self.light_curve.time.max() - self.light_curve.time.min()).to(u.day).value
        print(f"Time span: {time_span_days:.1f} days")
        print(f"Light curve time format: {self.light_curve.time.format}")
        print(f"Light curve time scale: {self.light_curve.time.scale}")
        print(f"First 5 time values (BKJD): {self.light_curve.time.value[:5]}")
        print(f"Transit epoch t0 (BKJD): {self.t0}")
        return self.light_curve
        
    def comprehensive_analysis(self):
        if self.light_curve is None:
            print("No data available for analysis")
            return
        fig = plt.figure(figsize=(16, 12))
        ax1 = plt.subplot(3, 2, 1)
        self.light_curve.plot(ax=ax1, color='blue', alpha=0.8)
        ax1.set_title('Kepler-10 Complete Light Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Flux')
        ax1.grid(True, alpha=0.3)
        ax2 = plt.subplot(3, 2, 2)
        mask_zoom = self.light_curve.time.value < (self.light_curve.time.value[0] + 20)
        zoom_lc = self.light_curve[mask_zoom]
        zoom_lc.plot(ax=ax2, color='red', alpha=0.8)
        ax2.set_title('Zoomed View (First 20 Days)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Normalized Flux')
        ax2.grid(True, alpha=0.3)
        ax3 = plt.subplot(3, 2, 3)
        folded_lc = self.light_curve.fold(period=self.period, epoch_time=self.t0)
        folded_lc.plot(ax=ax3, color='gray', alpha=0.3, markersize=1)
        ax3.set_title('Phase-Folded Transit (All Data)', fontsize=14, fontweight='bold')
        ax3.set_xlim(-0.1, 0.1)
        ax3.set_ylabel('Normalized Flux')
        ax3.grid(True, alpha=0.3)
        ax4 = plt.subplot(3, 2, 4)
        binned_lc = folded_lc.bin(time_bin_size=0.003)
        binned_lc.scatter(ax=ax4, c='red', s=15)
        ax4.set_title('Binned Phase-Folded Transit', fontsize=14, fontweight='bold')
        ax4.set_xlim(-0.03, 0.03)
        ax4.set_ylabel('Normalized Flux')
        ax4.grid(True, alpha=0.3)
        ax5 = plt.subplot(3, 2, 5)
        self.plot_individual_transits(ax5)
        ax6 = plt.subplot(3, 2, 6)
        self.transit_timing_analysis(ax6)
        plt.tight_layout()
        plt.show()
        self.detailed_measurements(folded_lc)
        
    def plot_individual_transits(self, ax):
        time_span_days = (self.light_curve.time.max() - self.light_curve.time.min()).to(u.day).value
        n_transits = int(time_span_days / self.period)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        plotted = 0
        for i in range(min(8, n_transits)):  
            transit_time = self.t0 + i * self.period
            if (transit_time < self.light_curve.time.value.min() or 
                transit_time > self.light_curve.time.value.max()):
                continue
            window = 0.15 
            mask = np.abs(self.light_curve.time.value - transit_time) < window
            if np.sum(mask) > 5: 
                transit_data = self.light_curve[mask]
                time_hours = (transit_data.time.value - transit_time) * 24
                ax.plot(time_hours, transit_data.flux, 
                       color=colors[plotted % len(colors)], 
                       alpha=0.7, linewidth=1,
                       label=f'Transit {plotted+1}')
                plotted += 1
        if plotted == 0:
            ax.text(0.5, 0.5, 'No individual transits found\nCheck time alignment or data coverage', 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_xlabel('Hours from Transit Center')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('Individual Transit Events', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        
    def transit_timing_analysis(self, ax):
        time_span_days = (self.light_curve.time.max() - self.light_curve.time.min()).to(u.day).value
        n_transits = int(time_span_days / self.period)
        observed_times = []
        transit_numbers = []
        for i in range(n_transits):
            expected_time = self.t0 + i * self.period
            if (expected_time < self.light_curve.time.value.min() or 
                expected_time > self.light_curve.time.value.max()):
                continue
            window = 0.1  
            mask = np.abs(self.light_curve.time.value - expected_time) < window
            if np.sum(mask) > 10:
                transit_data = self.light_curve[mask]
                min_idx = np.argmin(transit_data.flux)
                observed_time = transit_data.time.value[min_idx]
                observed_times.append(observed_time)
                transit_numbers.append(i)
        if len(observed_times) > 1:
            observed_times = np.array(observed_times)
            expected_times = np.array([self.t0 + i * self.period for i in transit_numbers])
            residuals = (observed_times - expected_times) * 24 * 60  
            ax.plot(transit_numbers, residuals, 'go-', markersize=6, linewidth=2)
            ax.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Transit Number')
            ax.set_ylabel('Timing Residual (minutes)')
            ax.set_title('Transit Timing Variations', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            rms = np.sqrt(np.mean(residuals**2))
            max_dev = np.max(np.abs(residuals))
            stats_text = f'RMS = {rms:.2f} min\nMax deviation = {max_dev:.2f} min'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient transits\nfor timing analysis', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            
    def detailed_measurements(self, folded_lc) :
        print("\n" + "="*60)
        print("KEPLER-10 TRANSIT ANALYSIS RESULTS")
        print("="*60)
        print(f"Target: {self.target_name}")
        print(f"Data points analyzed: {len(self.light_curve.time):,}")
        print(f"Time baseline: {(self.light_curve.time.max() - self.light_curve.time.min()).to(u.day).value:.1f} days")
        print(f"\nKnown Orbital Parameters:")
        print(f"Period: {self.period:.6f} days")
        print(f"Transit epoch: {self.t0:.2f} BKJD")
        transit_mask = np.abs(folded_lc.phase) < 0.015
        out_of_transit_mask = np.abs(folded_lc.phase) > 0.03
        if np.sum(transit_mask) > 0 and np.sum(out_of_transit_mask) > 0:
            in_transit_flux = np.median(folded_lc.flux[transit_mask])
            out_of_transit_flux = np.median(folded_lc.flux[out_of_transit_mask])
            transit_depth = (out_of_transit_flux - in_transit_flux) / out_of_transit_flux
            print(f"\nMeasured Transit Properties:")
            print(f"Transit depth: {transit_depth:.5f} ({transit_depth*100:.3f}%)")
            print(f"Transit duration: ~{self.period * 0.036 * 24:.2f} hours")
            if transit_depth > 0:
                radius_ratio = np.sqrt(transit_depth)
                print(f"Planet/Star radius ratio: {radius_ratio:.4f}")
                star_radius_km = 1.065 * 696340  
                planet_radius_km = radius_ratio * star_radius_km
                planet_radius_earth = planet_radius_km / 6371  
                print(f"Estimated planet radius: {planet_radius_earth:.2f} Earth radii")
                print(f"Estimated planet radius: {planet_radius_km:.0f} km")
        flux_std = np.std(self.light_curve.flux)
        print(f"\nData Quality:")
        print(f"Photometric precision: {flux_std*1e6:.0f} ppm")
        print(f"Number of transits in data: ~{int((self.light_curve.time.max() - self.light_curve.time.min()).to(u.day).value / self.period)}")
        print("="*60)
        print("Note: Kepler-10 b was the first confirmed rocky exoplanet discovered by Kepler!")
        print("="*60)

def main():
    print("Starting Kepler-10 exoplanet transit analysis...")
    print("Kepler-10 b: The first rocky exoplanet confirmed by Kepler")
    print("-" * 60)
    analyzer = Kepler10Analysis()
    light_curve = analyzer.download_and_process()
    if light_curve is not None:
        analyzer.comprehensive_analysis()
    else:
        print("Failed to download data. Please check your internet connection.")
if __name__ == "__main__":
    main()
