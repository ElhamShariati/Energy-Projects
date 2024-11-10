import numpy as np
import matplotlib.pyplot as plt

# Define fixed building parameters
area_per_floor = 225  # in square meters
number_of_floors = 2
window_count = 5
insulation_factor = 10  # arbitrary scale for insulation quality

# Define monthly environmental data for 12 months (example values)
solar_radiation = np.array([2.0, 2.7, 3.5, 4.5, 5.5, 5.9, 6.3, 5.9, 4.8, 3.6, 2.5, 2.0])  # kWh/m²/day
relative_humidity = np.array([77, 73, 66, 58, 56, 61, 64, 66, 68, 70, 76, 78])  # %
ambient_temperature = np.array([-14.9, -12.7, -5.3, 5.1, 12.6, 17.5, 20.2, 19.3, 13.5, 6.0, -4.9, -12.7])  # °C
wind_speed = np.array([16, 16, 18, 19, 19, 17, 16, 15, 16, 17, 17, 16])  # km/h

# Simulate actual energy consumption using all parameters
actual_consumption = (
    area_per_floor * number_of_floors * 0.5 +
    window_count * 1.5 +
    insulation_factor * 0.7 -
    solar_radiation * 5 +  # Solar energy reduces heating needs
    (20 - ambient_temperature) * 10 +  # Heating demand based on temperature difference
    wind_speed * 0.3 +  # Heat loss due to wind
    relative_humidity * 0.1 +  # Minor impact of humidity on energy use
    np.random.randn(12) * 5  # Adding some noise for realism
)

# Simulate predicted energy consumption (with minor deviations from actual values)
predicted_consumption = actual_consumption + np.random.randn(12) * 5

# Plotting the comparison
months = np.arange(1, 13)
plt.figure(figsize=(10, 6))
plt.plot(months, actual_consumption, label='Actual Energy Consumption', marker='o', color='blue')
plt.plot(months, predicted_consumption, label='Predicted Energy Consumption', marker='x', color='red')
plt.title('Comparison of Actual and Predicted Energy Consumption Over 12 Months')
plt.xlabel('Month')
plt.ylabel('Energy Consumption')
plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)
plt.show()
