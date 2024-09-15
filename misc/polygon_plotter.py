import matplotlib.pyplot as plt

# Extracting the x and y coordinates from the provided points
points = [
    (5772.62, 5683.07), (5776.56, 5683.97), (5779.07, 5685.41), (5780.21, 5687.83), 
    (5782.64, 5686.85), (5783.19, 5684.52), (5793.89, 5664.22), (5792.6, 5662.23), 
    (5788.5, 5662.19), (5788.58, 5659.3), (5786.93, 5655.8), (5785.73, 5653.29), 
    (5784.97, 5650.86), (5774.51, 5645.64), (5772.77, 5648.15), (5768.62, 5645.8), 
    (5768.01, 5650.64), (5764.91, 5650.22), (5761.58, 5650.6), (5759.32, 5652.06), 
    (5756.77, 5652.62), (5753.0, 5651.43), (5746.36, 5668.47), (5748.71, 5669.41), 
    (5745.96, 5673.03), (5747.88, 5674.32), (5751.36, 5672.8), (5753.29, 5676.55), 
    (5755.34, 5679.39), (5756.4, 5681.28), (5756.78, 5683.37), (5771.25, 5686.89), 
    (5772.62, 5683.07)
]

# Separate the x and y coordinates
x_vals, y_vals = zip(*points)

# Plot the polygon
plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, 'b-', marker='o')
plt.fill(x_vals, y_vals, 'lightblue', alpha=0.5)  # Fill the polygon
plt.title('Convex Polygon Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
