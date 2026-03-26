import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']  # Use English font
plt.rcParams['axes.unicode_minus'] = False    # Display minus sign normally
plt.rcParams['figure.figsize'] = (10, 6)      # Fixed figure size (width=10, height=6)
plt.rcParams['savefig.dpi'] = 300             # High resolution (300 dpi) for saved images

# ===================== 2. Evaluation Metrics Function =====================
def calculate_metrics(y_true, y_pred):
    """Calculate R-squared (R²) and Root Mean Squared Error (RMSE)"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0
    return round(r2, 4), round(rmse, 4)

# ===================== 3. Linear Regression Class (3 Methods) =====================
class LinearRegression:
    def __init__(self):
        self.coef_ = None  # Slope (β1)
        self.intercept_ = None  # Intercept (β0)

    # 3.1 Least Squares (Analytical Solution)
    def fit_least_squares(self, X, y):
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias column for β0
        beta = pinv(X_aug.T @ X_aug) @ X_aug.T @ y  # Use pseudoinverse to avoid singularity
        self.intercept_, self.coef_ = beta[0], beta[1:]

    # 3.2 Gradient Descent (Iterative Optimization)
    def fit_gradient_descent(self, X, y, lr=0.01, epochs=10000, tol=1e-6):
        n_samples = X.shape[0]
        self.intercept_, self.coef_ = 0.0, np.zeros(1)  # Initialize parameters
        prev_loss = float('inf')  # Loss from previous iteration

        for _ in range(epochs):
            y_pred = self.intercept_ + X @ self.coef_
            current_loss = np.mean((y - y_pred) ** 2)

            # Stop if converged (loss change < tolerance)
            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

            # Calculate gradients
            grad_intercept = -2 * np.mean(y - y_pred)
            grad_coef = -2 * np.mean((y - y_pred).reshape(-1, 1) * X)

            # Update parameters
            self.intercept_ -= lr * grad_intercept
            self.coef_ -= lr * grad_coef

    # 3.3 Newton's Method (Second-Order Optimization)
    def fit_newton(self, X, y, tol=1e-6, max_iter=100):
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias column
        theta = np.zeros(2)  # Initialize parameters [β0, β1]

        for _ in range(max_iter):
            y_pred = X_aug @ theta
            residual = y_pred - y

            # Calculate gradient (1st derivative) and Hessian (2nd derivative)
            grad = (1 / len(X)) * X_aug.T @ residual
            hessian = (1 / len(X)) * X_aug.T @ X_aug

            # Update parameters using Newton's rule
            delta = pinv(hessian) @ grad
            theta -= delta

            # Stop if converged (parameter change < tolerance)
            if np.linalg.norm(delta) < tol:
                break

        self.intercept_, self.coef_ = theta[0], theta[1:]

    # Prediction Function
    def predict(self, X):
        """Generate predictions using trained parameters"""
        return self.intercept_ + X @ self.coef_

# ===================== 4. Visualization Function (Plot Data + Fitting Line) =====================
def plot_fitting(method_name, X, y, y_pred, intercept, coef, data_type, save_path):
    """
    Plot scatter (data points) and line (fitting result)
    :param method_name: Name of fitting method (e.g., "Least Squares")
    :param X: Feature data (x)
    :param y: True target values (y)
    :param y_pred: Predicted values (fitted y)
    :param intercept: Model intercept (β0)
    :param coef: Model slope (β1)
    :param data_type: Type of data ("Training Set" or "Test Set")
    :param save_path: Path to save the plot
    """
    # Sort X and y_pred to ensure a smooth fitting line
    sorted_idx = np.argsort(X.flatten())
    X_sorted = X[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot data points (blue scatter, semi-transparent)
    ax.scatter(X, y, color='#1f77b4', alpha=0.6, s=50, label=f'{data_type} Data')

    # Plot fitting line (orange solid line, bold)
    ax.plot(X_sorted, y_pred_sorted, color='#ff7f0e', linewidth=2.5,
            label=f'Fitting Line: y = {intercept:.4f} + {coef[0]:.4f}x')

    # Calculate and display metrics (R², RMSE) in top-left corner
    r2, rmse = calculate_metrics(y, y_pred)
    ax.text(0.05, 0.95, f'R-squared: {r2}\nRMSE: {rmse}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))  # White background for readability

    # Set plot title and axis labels
    ax.set_title(f'{method_name} - {data_type} Fitting Result', fontsize=14, pad=20)
    ax.set_xlabel('X (Independent Variable)', fontsize=12)
    ax.set_ylabel('Y (Dependent Variable)', fontsize=12)

    # Add legend and grid
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)  # Light gray grid (non-intrusive)

    # Save plot (adjust layout to avoid label cutoff)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close figure to free memory

# ===================== 5. Main Function (Full Workflow) =====================
def main(train_path, test_path):
    # 5.1 Load Training and Test Data
    try:
        train_df = pd.read_excel(train_path, header=0)
        test_df = pd.read_excel(test_path, header=0)

        # Extract features (x, 1st column) and target (y, 2nd column)
        X_train = train_df.iloc[:, 0].values.reshape(-1, 1)
        y_train = train_df.iloc[:, 1].values
        X_test = test_df.iloc[:, 0].values.reshape(-1, 1)
        y_test = test_df.iloc[:, 1].values

        print("Data loaded successfully!")
        print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    except Exception as e:
        print(f"Data load failed: {str(e)}")
        return

    # 5.2 Initialize Model and Fitting Methods
    model = LinearRegression()
    methods = {
        "Least Squares": model.fit_least_squares,
        "Gradient Descent": model.fit_gradient_descent,
        "Newton's Method": model.fit_newton
    }

    # 5.3 Train, Predict, and Plot for Each Method
    for method_name, fit_func in methods.items():
        print(f"\nProcessing {method_name}...")

        # Train the model
        fit_func(X_train, y_train)

        # Generate predictions for training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Get current model parameters (intercept, slope)
        current_intercept = model.intercept_
        current_coef = model.coef_

        # Plot and save Training Set fitting result
        train_plot_path = f"./{method_name.replace(' ', '_')}_Training.png"
        plot_fitting(method_name, X_train, y_train, y_train_pred,
                    current_intercept, current_coef, "Training Set", train_plot_path)

        # Plot and save Test Set fitting result
        test_plot_path = f"./{method_name.replace(' ', '_')}_Test.png"
        plot_fitting(method_name, X_test, y_test, y_test_pred,
                    current_intercept, current_coef, "Test Set", test_plot_path)

        print(f"{method_name} plots saved: {train_plot_path} | {test_plot_path}")

# ===================== 6. Execution Entry =====================
if __name__ == "__main__":
    # Replace with your actual Excel file paths (use r-prefix to avoid escape characters)
    TRAIN_FILE = r"C:\Users\ALIENWARE\Desktop\PRML\h1\Data4Regression.xlsx"
    TEST_FILE = r"C:\Users\ALIENWARE\Desktop\PRML\h1\Data4Regression_test.xlsx"

    # Run the main workflow
    main(TRAIN_FILE, TEST_FILE)