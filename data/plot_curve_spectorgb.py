import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('spec_to_rgb.csv')

df[['R', 'G', 'B']] = df[['R', 'G', 'B']] / 255.0

plt.figure(figsize=(12, 6))
plt.plot(df['wavelength'], df['R'], color='red', label='R')
plt.plot(df['wavelength'], df['G'], color='green', label='G')
plt.plot(df['wavelength'], df['B'], color='blue', label='B')

plt.title('Wavelength vs RGB Values')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized RGB Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, 'wavelength_rgb_plot.png'), dpi=300, bbox_inches='tight')

print(f"Plot saved as 'wavelength_rgb_plot.png' in {script_dir}")