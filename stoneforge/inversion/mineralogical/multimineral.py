import numpy as np
import pandas as pd
import json

class MineralDatabase:
    """Handles loading and parsing of petrophysical mineral endpoint properties

    from various file formats (CSV, JSON, or Dictionary).
    """

    def __init__(self, data_source=None):
        self.matrix = {}
        if data_source:
            if isinstance(data_source, dict):
                self.load_from_dict(data_source)
            elif isinstance(data_source, str):
                if data_source.endswith(".csv"):
                    self.load_from_csv(data_source)
                elif data_source.endswith(".json"):
                    self.load_from_json(data_source)

    def load_from_dict(self, d: dict):
        """Loads database from a standard python dictionary."""
        self.matrix = {
            str(min_k).upper(): {str(log_k).upper(): float(val) for log_k, val in log_v.items()}
            for min_k, log_v in d.items()
        }

    def load_from_json(self, filepath: str):
        """Loads database from a JSON file."""
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        self.load_from_dict(raw_data)

    def load_from_csv(self, filepath: str):
        """Loads database from a standard CSV file."""
        df = pd.read_csv(filepath)
        # Ensure the first column contains the mineral/element identifier string
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(str).str.upper()
        df = df.set_index(first_col)

        # Convert dataframe directly to the nested dictionary configuration format
        raw_data = df.to_dict(orient="index")
        self.load_from_dict(raw_data)

    def get_matrix(self) -> dict:
        return self.matrix


class MultimineralInversion:
    """Highly optimized, loop-free Multimineral Linear Inversion Engine."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.df.columns = [str(col).upper() for col in self.df.columns]
        self.num_samples = len(self.df)

        # Inject the volume constraint parameter (Sum of Volumes = 1)
        self.df["ONES"] = np.ones(self.num_samples)

    def solve(
        self,
        db: MineralDatabase,
        active_logs: list,
        active_minerals: list,
        method: str = "least_squares",
        reg: float = 0.0,
        enforce_positive: bool = True,
    ) -> pd.DataFrame:
        """Dynamically builds matrix mappings based on selected logs/minerals and solves."""
        active_logs = [str(log).upper() for log in active_logs]
        active_minerals = [str(m).upper() for m in active_minerals]
        mineral_matrix = db.get_matrix()

        # Build local copies for constraint logic
        solver_logs = list(active_logs)
        if "ONES" not in solver_logs:
            solver_logs.append("ONES")

        # 1. Dynamically build Response Matrix (A) based strictly on active selections
        # Shape: (M logs, N minerals)
        try:
            A = []
            for log in solver_logs:
                row = []
                for mineral in active_minerals:
                    # Default to 1.0 if checking ONES constraint, else pull from DB
                    val = 1.0 if log == "ONES" else mineral_matrix[mineral][log]
                    row.append(val)
                A.append(row)
            A = np.array(A, dtype=float)
        except KeyError as e:
            raise KeyError(
                f"Requested pair not found in database. Check logs/mineral definitions: {e}"
            )

        # 2. Extract Data Matrix (B) from DataFrame -> Shape: (M logs, Depth Samples)
        B = self.df[solver_logs].to_numpy().T

        # 3. Regularization term
        identity_reg = np.eye(A.shape[1] if method != "exact" else A.shape[0]) * reg

        # 4. Solvers
        if method == "exact":
            A_inv = np.linalg.inv(A + identity_reg)
            X = np.dot(A_inv, B)
        elif method == "least_squares":
            ATA = np.dot(A.T, A)
            A_inv = np.linalg.inv(ATA + identity_reg)
            X = np.dot(np.dot(A_inv, A.T), B)
        elif method == "underdetermined":
            AAT = np.dot(A, A.T)
            A_inv = np.linalg.inv(AAT + identity_reg)
            X = np.dot(np.dot(A.T, A_inv), B)
        else:
            raise ValueError(f"Unknown method: {method}")

        X = X.T  # Transpose to shape: (Depth Samples, N minerals)

        # 5. Volumetric Adjustments (Non-negative & Unity Constraint closure)
        if enforce_positive:
            X = np.clip(X, a_min=0, a_max=None)
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums

        return pd.DataFrame(X, columns=active_minerals, index=self.df.index)