import numpy as np
import cv2
import matplotlib.pyplot as plt

# Class for calculating gradient magnitudes and squared gradients
class GradientMagnitude:
    def __init__(self, image_path, sigma=1, kernel_size=5):
        """Initialize with image path, Gaussian sigma, and kernel size."""
        self.image_path = image_path
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.image is None:
            raise ValueError("Image not found or could not be loaded.")

        self.gaussian_kernel = None
        self.gradient_x = None
        self.gradient_y = None
        self.I_x = None
        self.I_y = None
        self.gradient_magnitude = None
        self.squared_gradient_magnitude = None

    def gaussian_kernel_and_gradients(self):
        """Generate Gaussian kernel and its gradients."""
        ax = np.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        g = np.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma ** 2))
        g /= g.sum()  # Normalize
        self.gaussian_kernel = g

        # Gradient in x and y directions
        self.gradient_x = -xx * g / (2 * np.pi * self.sigma ** 4)
        self.gradient_y = -yy * g / (2 * np.pi * self.sigma ** 4)

    def compute_squared_gradient_magnitude(self):
        """Compute the squared gradient magnitude of the image using Gaussian gradient."""
        self.gaussian_kernel_and_gradients()

        # Convolve image with gradients
        self.I_x = cv2.filter2D(self.image, -1, self.gradient_x)
        self.I_y = cv2.filter2D(self.image, -1, self.gradient_y)

        # Compute gradient magnitude
        self.gradient_magnitude = np.sqrt(self.I_x ** 2 + self.I_y ** 2)

        # Square the gradient magnitude
        self.squared_gradient_magnitude = self.gradient_magnitude ** 2

        return self.squared_gradient_magnitude


class ActiveContour:
    def __init__(self, image, n_points, alpha=1.0, beta=1.0, gamma=1.0):
        """Initialize with image, number of contour points, and energy weights."""
        self.image = image
        self.n_points = n_points
        self.alpha = alpha  # Weight for elastic energy
        self.beta = beta  # Weight for smooth energy
        self.gamma = gamma  # Weight for attraction energy
        self.contour_points = self.calculate_contour_points()
        self.original_contour_points = self.contour_points.copy()  # Save original contour points

    def calculate_contour_points(self):
        """Calculate contour points in a circular manner around the center of the image."""
        height, width = self.image.shape
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y) - 10  # Adjusted to increase the initial contour size
  # Adjust radius to fit within the image
        angles = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        contour_points = [(int(center_x + radius * np.cos(angle)), int(center_y + radius * np.sin(angle))) for angle in
                          angles]
        return contour_points

    def calculate_total_energy(self, squared_gradient_magnitude_image):
        """Calculate the energy E at the current contour points."""
        energy = 0.0
        for i, (x, y) in enumerate(self.contour_points):
            # Ensure the point is within bounds of the image
            if 0 <= x < squared_gradient_magnitude_image.shape[1] and 0 <= y < squared_gradient_magnitude_image.shape[0]:
                total_energy = self.calculate_energy_at_point(i, x, y, squared_gradient_magnitude_image)
                energy += total_energy  # Accumulate the total energy
        return energy

    def calculate_energy_at_point(self, index, x, y, squared_gradient_magnitude_image):
        """Calculate the total energy at a specific point."""
        gradient_energy = self.calculate_image_energy(x, y, squared_gradient_magnitude_image)
        elastic_energy = self.calculate_elastic_energy(index)
        smooth_energy = self.calculate_smooth_energy(index)
        attraction_energy = -self.calculate_attraction_energy(x, y)
        total_energy = gradient_energy + self.alpha * elastic_energy + self.beta * smooth_energy + self.gamma * attraction_energy
        return -total_energy

    def calculate_elastic_energy(self, index):
        """Calculate the elastic energy for a contour point."""
        prev_point = self.contour_points[index - 1]
        curr_point = self.contour_points[index]
        next_point = self.contour_points[(index + 1) % self.n_points]
        elastic_energy = np.linalg.norm(np.array(curr_point) - np.array(prev_point)) + np.linalg.norm(np.array(curr_point) - np.array(next_point))
        return elastic_energy

    def calculate_smooth_energy(self, index):
        """Calculate the smooth energy for a contour point."""
        prev_point = self.contour_points[index - 1]
        curr_point = self.contour_points[index]
        next_point = self.contour_points[(index + 1) % self.n_points]
        smooth_energy = np.linalg.norm(np.array(next_point) - 2 * np.array(curr_point) + np.array(prev_point))
        return smooth_energy

    def calculate_image_energy(self, x, y, squared_gradient_magnitude_image):
        """Calculate the energy at a specific point in the image."""
        return squared_gradient_magnitude_image[y, x]

    def calculate_attraction_energy(self, x, y):
        """Calculate the attraction energy to pull points towards the center."""
        height, width = self.image.shape
        center_x, center_y = width // 2, height // 2
        attraction_energy = np.linalg.norm(np.array([x, y]) - np.array([center_x, center_y]))
        return attraction_energy

    def move_contour_points(self, squared_gradient_magnitude_image, step_size=1):
        """Move contour points to minimize the energy."""
        new_contour_points = []
        for i, (x, y) in enumerate(self.contour_points):
            # Initialize with the current point energy
            min_energy = self.calculate_energy_at_point(i, x, y, squared_gradient_magnitude_image)
            best_point = (x, y)

            for dx in range(-step_size, step_size + 1):
                for dy in range(-step_size, step_size + 1):
                    new_x = x + dx
                    new_y = y + dy

                    # Check bounds
                    if 0 <= new_x < squared_gradient_magnitude_image.shape[1] and 0 <= new_y < squared_gradient_magnitude_image.shape[0]:
                        # Calculate energy for the new point
                        new_energy = self.calculate_energy_at_point(i, new_x, new_y, squared_gradient_magnitude_image)
                        if new_energy < min_energy:
                            min_energy = new_energy
                            best_point = (new_x, new_y)

            new_contour_points.append(best_point)

        self.contour_points = new_contour_points  # Update contour points with new positions

    def optimize_contour(self, squared_gradient_magnitude_image, iterations=100, step_size=1, threshold=1e-4):
        """Iterate to minimize the energy by moving contour points. Stop if energy change in last 5 iterations is small."""

        # Setup a live plot for displaying iterations
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(6, 6))

        # List to track the energies over the last 5 iterations
        energy_history = []

        for i in range(iterations):
            print(f"Iteration {i + 1}/{iterations}")

            # Calculate the energy at this iteration
            energy = self.calculate_total_energy(squared_gradient_magnitude_image)
            print(f"Energy at iteration {i + 1}: {energy}")

            # Add current energy to the history and keep only the last 5 energies
            energy_history.append(energy)
            if len(energy_history) > 5:
                energy_history.pop(0)

            # Check if the difference between energies is below the threshold for the last 5 iterations
            if len(energy_history) == 5 and max(energy_history) - min(energy_history) < threshold:
                print(f"Terminating early: change in energy over the last 5 iterations is below {threshold}")
                break  # Terminate the optimization if the change is small enough

            # Move contour points to minimize energy
            self.move_contour_points(squared_gradient_magnitude_image, step_size)

            # Clear the plot and update it with the new contours
            ax.clear()
            self.show_image_with_contours(squared_gradient_magnitude_image, ax=ax)
            plt.draw()
            plt.pause(0.5)  # Pause for a moment before updating the plot

        # Turn off interactive mode after optimization
        plt.ioff()
        plt.show()

        # Calculate and print the final energy after optimization
        final_energy = self.calculate_total_energy(squared_gradient_magnitude_image)
        print(f"Final Energy after optimization: {final_energy}")

        # Turn off interactive mode after optimization
        plt.ioff()
        plt.show()

    def show_image_with_contours(self, squared_gradient_magnitude_image, ax=None):
        """Display the image with both original and updated contour points marked."""

        image_with_contours = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Draw original contour points in red
        for point in self.original_contour_points:
            cv2.circle(image_with_contours, point, 1, (255, 0, 0), -1)  # Red

        # Draw updated contour points in green
        for point in self.contour_points:
            cv2.circle(image_with_contours, point, 1, (0, 255, 0), -1)  # Green

        # Convert to RGB for matplotlib and display in the passed axis
        ax.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        ax.set_title('Image with Original and Updated Contour Points')
        ax.axis('off')


#
# Path to your image
image_path = 'sample.png'  # Change to the path of your image

# 1. Create an instance of GradientMagnitude
gradient_magnitude_calculator = GradientMagnitude(image_path, sigma=1, kernel_size=15)

# 2. Compute the squared gradient magnitude of the image
squared_gradient_magnitude_image = gradient_magnitude_calculator.compute_squared_gradient_magnitude()

# Convert the squared gradient magnitude to float32 (required for some operations)
squared_gradient_magnitude_image = squared_gradient_magnitude_image.astype(np.float32)

# Display the squared gradient magnitude image
plt.figure(figsize=(6, 6))
plt.title("Squared Gradient Magnitude")
plt.imshow(squared_gradient_magnitude_image, cmap='gray')
plt.axis('off')
plt.show()

# 3. Create an instance of ActiveContour
n_points = 1000  # Number of points on the contour
active_contour = ActiveContour(gradient_magnitude_calculator.image, n_points, alpha=1, beta=1.0, gamma=1.0)

# 4. Calculate the initial energy of the contour
initial_energy = active_contour.calculate_total_energy(squared_gradient_magnitude_image)
print(f"Initial Energy: {initial_energy}")

# 5. Optimize contour points to minimize the energy
# This will iterate and adjust the contour points
active_contour.optimize_contour(squared_gradient_magnitude_image, iterations=100, step_size=1)

# Final energy will be displayed at the end of the optimization.
