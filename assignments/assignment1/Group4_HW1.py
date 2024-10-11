import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self):
        self.X_Train = None
        self.Y_Train = None
        self.X_Test = None
        self.Y_Test = None

    def load_data(self):
        # Load the dataset
        self.X_Train = np.array([
            [1.3, 3.3], [1.4, 2.5], [1.8, 2.8], [1.9, 3.1],
            [1.5, 1.5], [1.8, 2], [2.3, 1.9], [2.4, 1.4],
            [2.4, 2.4], [2.4, 3], [2.7, 2.7], [2.3, 3.2]
        ])
        self.Y_Train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        self.X_Test = np.array([[1.7, 2.5], [1.9, 2.7], [2, 2.15], [2.4, 2], [2.2, 3.25], [2.4, 2.25]])
        self.Y_Test = np.array([0, 0, 1, 1, 2, 2])


# Part 1: Default SVM Implementation and Analysis
class SVMClassifier:
    def __init__(self, data_loader, C=1.0):
        # Initialize the SVM classifier with default parameters
        self.clf = svm.SVC(C=C)
        self.data_loader = data_loader  # Reference to the DataLoader instance

    def train(self):
        # Train the SVM classifier
        self.clf.fit(self.data_loader.X_Train, self.data_loader.Y_Train)

    def predict(self):
        # Predict on training and test data
        self.Y_Pred_Train = self.clf.predict(self.data_loader.X_Train)
        self.Y_Pred_Test = self.clf.predict(self.data_loader.X_Test)

    def get_confusion_matrices(self):
        # Obtain confusion matrices
        self.cm_train = confusion_matrix(self.data_loader.Y_Train, self.Y_Pred_Train)
        self.cm_test = confusion_matrix(self.data_loader.Y_Test, self.Y_Pred_Test)
        return self.cm_train, self.cm_test

    def visualize_decision_surface(self):
        # Visualization of decision surfaces
        markers_train = ['o', 's', '^']
        markers_test = ['*', '*', '*']
        colors = ['blue', 'red', 'orange']
        cmap = ListedColormap(colors)

        X_Train = self.data_loader.X_Train
        Y_Train = self.data_loader.Y_Train
        X_Test = self.data_loader.X_Test
        Y_Test = self.data_loader.Y_Test

        """
        
        1. X_Train[:, 0] and X_Train[:, 1] extract the first and second features (dimensions) 
        from the training data, respectively.
        
        2. .min() and .max() find the minimum and maximum values of these features.
        
        3. Padding: Subtracting 1 from the minimum and adding 1 to the maximum values to ensure 
        that the decision boundary is fully visible and not cut off at the edges.
        
        """
        # Step size in the mesh, determining the resolution of the grid
        h = .02
        x_min, x_max = X_Train[:, 0].min() - 1, X_Train[:, 0].max() + 1
        y_min, y_max = X_Train[:, 1].min() - 1, X_Train[:, 1].max() + 1
        # Create a grid over the feature space
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        # Predict over meshgrid
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

        # Plot training points
        for idx, cl in enumerate(np.unique(Y_Train)):
            plt.scatter(
                X_Train[Y_Train == cl][:, 0],
                X_Train[Y_Train == cl][:, 1],
                c=colors[idx], marker=markers_train[idx],
                label=f'Train Class {cl}', edgecolor='k', s=100
            )

        # Plot test points
        for idx, cl in enumerate(np.unique(Y_Test)):
            plt.scatter(
                X_Test[Y_Test == cl][:, 0],
                X_Test[Y_Test == cl][:, 1],
                c=colors[idx], marker=markers_test[idx],
                label=f'Test Class {cl}', edgecolor='k', s=200
            )

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Surface of SVM')
        plt.legend()
        plt.show()

    @classmethod
    def showcase(cls):
        # Instantiate the DataLoader and load data
        data_loader = DataLoader()
        data_loader.load_data()

        # Instantiate the SVMClassifier with the DataLoader
        svm_classifier = cls(data_loader)

        # Train the model
        svm_classifier.train()

        # Predict on training and test data
        svm_classifier.predict()

        # Get confusion matrices
        cm_train, cm_test = svm_classifier.get_confusion_matrices()

        # Display confusion matrices
        print("Confusion Matrix for Training Data:")
        print(cm_train)
        print("\nConfusion Matrix for Test Data:")
        print(cm_test)

        # Visualize decision surfaces
        svm_classifier.visualize_decision_surface()


# Part 2: One-vs-Rest SVM and Perceptron Analysis
class OneVsRestClassifierPart2:
    def __init__(self, data_loader):
        self.data_loader = data_loader  # Reference to the DataLoader instance
        self.classes = np.unique(self.data_loader.Y_Train)
        self.metrics = {}

    def prepare_labels(self, class_label):
        # Prepare binary labels for one-vs-rest classification
        self.Y_Train_binary = np.where(self.data_loader.Y_Train == class_label, 1, 0)
        self.Y_Test_binary = np.where(self.data_loader.Y_Test == class_label, 1, 0)

    def train_models(self):
        # Train SVM and Perceptron models
        self.svm_model = svm.SVC(C=1.0)
        self.perceptron_model = Perceptron(max_iter=1000, tol=1e-3)

        self.svm_model.fit(self.data_loader.X_Train, self.Y_Train_binary)
        self.perceptron_model.fit(self.data_loader.X_Train, self.Y_Train_binary)

    def predict(self):
        # Predict using SVM and Perceptron models
        self.Y_Pred_SVM_Train = self.svm_model.predict(self.data_loader.X_Train)
        self.Y_Pred_SVM_Test = self.svm_model.predict(self.data_loader.X_Test)

        self.Y_Pred_Perc_Train = self.perceptron_model.predict(self.data_loader.X_Train)
        self.Y_Pred_Perc_Test = self.perceptron_model.predict(self.data_loader.X_Test)

    def compute_metrics(self, class_label):
        # Compute performance metrics
        metrics = {}
        for model_name, Y_Pred_Train, Y_Pred_Test in [
            ('SVM', self.Y_Pred_SVM_Train, self.Y_Pred_SVM_Test),
            ('Perceptron', self.Y_Pred_Perc_Train, self.Y_Pred_Perc_Test)
        ]:
            metrics[model_name] = {
                'Accuracy_Train': accuracy_score(self.Y_Train_binary, Y_Pred_Train),
                'Precision_Train': precision_score(self.Y_Train_binary, Y_Pred_Train, zero_division=0),
                'Recall_Train': recall_score(self.Y_Train_binary, Y_Pred_Train, zero_division=0),
                'F1_Train': f1_score(self.Y_Train_binary, Y_Pred_Train, zero_division=0),
                'Accuracy_Test': accuracy_score(self.Y_Test_binary, Y_Pred_Test),
                'Precision_Test': precision_score(self.Y_Test_binary, Y_Pred_Test, zero_division=0),
                'Recall_Test': recall_score(self.Y_Test_binary, Y_Pred_Test, zero_division=0),
                'F1_Test': f1_score(self.Y_Test_binary, Y_Pred_Test, zero_division=0),
                'Confusion_Matrix_Train': confusion_matrix(self.Y_Train_binary, Y_Pred_Train),
                'Confusion_Matrix_Test': confusion_matrix(self.Y_Test_binary, Y_Pred_Test)
            }
        self.metrics[class_label] = metrics

    def visualize_decision_boundaries(self, class_label):
        # Visualization of decision boundaries
        colors = ['blue', 'red']
        cmap = ListedColormap(colors)
        markers = ['o', 's']

        h = .02  # Step size in the mesh
        X_Train = self.data_loader.X_Train
        Y_Train_binary = self.Y_Train_binary
        X_Test = self.data_loader.X_Test
        Y_Test_binary = self.Y_Test_binary

        x_min, x_max = X_Train[:, 0].min() - 1, X_Train[:, 0].max() + 1
        y_min, y_max = X_Train[:, 1].min() - 1, X_Train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        models = {
            'SVM': self.svm_model,
            'Perceptron': self.perceptron_model
        }

        for model_name, model in models.items():
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(10, 6))
            plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

            # Plot training points
            for idx, label in enumerate(np.unique(Y_Train_binary)):
                plt.scatter(X_Train[Y_Train_binary == label, 0],
                            X_Train[Y_Train_binary == label, 1],
                            c=colors[idx], marker=markers[idx],
                            label=f'Train Class {label}', edgecolor='k', s=100)

            # Plot test points
            for idx, label in enumerate(np.unique(Y_Test_binary)):
                plt.scatter(X_Test[Y_Test_binary == label, 0],
                            X_Test[Y_Test_binary == label, 1],
                            c=colors[idx], marker='*',
                            label=f'Test Class {label}', edgecolor='k', s=200)

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Decision Boundary of {model_name} for Class {class_label} vs Rest')
            plt.legend()
            plt.show()

    @classmethod
    def showcase(cls):
        # Instantiate the DataLoader and load data
        data_loader = DataLoader()
        data_loader.load_data()

        # Instantiate the OneVsRestClassifierPart2 with the DataLoader
        ovr_classifier = cls(data_loader)

        # For each class, perform one-vs-rest classification
        for class_label in ovr_classifier.classes:
            print(f"\n--- Class {class_label} vs Rest ---")

            # Prepare binary labels
            ovr_classifier.prepare_labels(class_label)

            # Train models
            ovr_classifier.train_models()

            # Predict
            ovr_classifier.predict()

            # Compute metrics
            ovr_classifier.compute_metrics(class_label)

            # Display metrics
            for model_name in ['SVM', 'Perceptron']:
                print(f"\n{model_name} Metrics:")

                metrics = ovr_classifier.metrics[class_label][model_name]
                print(f"Accuracy (Train): {metrics['Accuracy_Train']:.2f}")
                print(f"Precision (Train): {metrics['Precision_Train']:.2f}")
                print(f"Recall (Train): {metrics['Recall_Train']:.2f}")
                print(f"F1 Score (Train): {metrics['F1_Train']:.2f}\n")

                print(f"Accuracy (Test): {metrics['Accuracy_Test']:.2f}")
                print(f"Precision (Test): {metrics['Precision_Test']:.2f}")
                print(f"Recall (Test): {metrics['Recall_Test']:.2f}")
                print(f"F1 Score (Test): {metrics['F1_Test']:.2f}\n")

                print("Confusion Matrix (Train):")
                print(metrics['Confusion_Matrix_Train'])
                print("Confusion Matrix (Test):")
                print(metrics['Confusion_Matrix_Test'])

            # Visualize decision boundaries
            ovr_classifier.visualize_decision_boundaries(class_label)


class OneVsRestClassifierPart2WithNoise(OneVsRestClassifierPart2):
    def add_noise(self, noise_level=0.05):
        # Flip labels for a percentage of training data
        num_samples = len(self.data_loader.Y_Train)
        num_noisy = max(1, int(noise_level * num_samples))
        indices = np.arange(num_samples)
        np.random.seed(42)  # For reproducibility
        noisy_indices = np.random.choice(indices, size=num_noisy, replace=False)
        self.Y_Train_noisy = self.data_loader.Y_Train.copy()
        for idx in noisy_indices:
            # Flip label to a random different class
            current_label = self.Y_Train_noisy[idx]
            possible_labels = [l for l in np.unique(self.Y_Train_noisy) if l != current_label]
            self.Y_Train_noisy[idx] = np.random.choice(possible_labels)
        print(f"Added noise to indices: {noisy_indices}")
        # Update the DataLoader's Y_Train with noisy labels
        self.data_loader.Y_Train = self.Y_Train_noisy

    @classmethod
    def showcase(cls):
        # Instantiate the DataLoader and load data
        data_loader = DataLoader()
        data_loader.load_data()

        # Instantiate the OneVsRestClassifierWithNoise with the DataLoader
        ovr_classifier_noise = cls(data_loader)

        # Add noise to the training labels
        ovr_classifier_noise.add_noise(noise_level=0.05)

        # For each class, perform one-vs-rest classification with noise
        for class_label in ovr_classifier_noise.classes:
            print(f"\n--- Class {class_label} vs Rest with Noise ---")

            # Prepare binary labels
            ovr_classifier_noise.prepare_labels(class_label)

            # Train models
            ovr_classifier_noise.train_models()

            # Predict
            ovr_classifier_noise.predict()

            # Compute metrics
            ovr_classifier_noise.compute_metrics(class_label)

            # Display metrics
            for model_name in ['SVM', 'Perceptron']:
                print(f"\n{model_name} Metrics with Noise:")

                metrics = ovr_classifier_noise.metrics[class_label][model_name]
                print(f"Accuracy (Train): {metrics['Accuracy_Train']:.2f}")
                print(f"Precision (Train): {metrics['Precision_Train']:.2f}")
                print(f"Recall (Train): {metrics['Recall_Train']:.2f}")
                print(f"F1 Score (Train): {metrics['F1_Train']:.2f}\n")

                print(f"Accuracy (Test): {metrics['Accuracy_Test']:.2f}")
                print(f"Precision (Test): {metrics['Precision_Test']:.2f}")
                print(f"Recall (Test): {metrics['Recall_Test']:.2f}")
                print(f"F1 Score (Test): {metrics['F1_Test']:.2f}\n")

                print("Confusion Matrix (Train):")
                print(metrics['Confusion_Matrix_Train'])
                print("Confusion Matrix (Test):")
                print(metrics['Confusion_Matrix_Test'])

            # Visualize decision boundaries
            ovr_classifier_noise.visualize_decision_boundaries(class_label)


# Part 3: Aggregated Results from One-vs-Rest Strategy
class OneVsRestClassifierPart3:
    def __init__(self, data_loader):
        self.data_loader = data_loader  # Reference to the DataLoader instance
        self.classes = np.unique(self.data_loader.Y_Train)
        self.models = {}
        self.metrics = {}
        self.decision_functions_train = {}
        self.decision_functions_test = {}

    def train_models(self, C=1.0):
        # Train SVM and Perceptron models for each class
        self.svm_models = {}
        self.perceptron_models = {}
        for class_label in self.classes:
            # Prepare binary labels
            Y_Train_binary = np.where(self.data_loader.Y_Train == class_label, 1, 0)
            svm_model = svm.SVC(C=C, decision_function_shape='ovr')
            perceptron_model = Perceptron(max_iter=1000, tol=1e-3)

            svm_model.fit(self.data_loader.X_Train, Y_Train_binary)
            perceptron_model.fit(self.data_loader.X_Train, Y_Train_binary)

            self.svm_models[class_label] = svm_model
            self.perceptron_models[class_label] = perceptron_model

    def predict(self):
        # Aggregate predictions
        # SVM
        decision_values_svm_train = []
        decision_values_svm_test = []
        for class_label in self.classes:
            svm_model = self.svm_models[class_label]
            # Get decision function values
            decision_train = svm_model.decision_function(self.data_loader.X_Train)
            decision_test = svm_model.decision_function(self.data_loader.X_Test)
            decision_values_svm_train.append(decision_train)
            decision_values_svm_test.append(decision_test)

        decision_values_svm_train = np.vstack(decision_values_svm_train).T
        decision_values_svm_test = np.vstack(decision_values_svm_test).T

        self.Y_Pred_SVM_Train = self.classes[np.argmax(decision_values_svm_train, axis=1)]
        self.Y_Pred_SVM_Test = self.classes[np.argmax(decision_values_svm_test, axis=1)]

        # Perceptron
        decision_values_perc_train = []
        decision_values_perc_test = []
        for class_label in self.classes:
            perceptron_model = self.perceptron_models[class_label]
            # Get decision function values
            decision_train = perceptron_model.decision_function(self.data_loader.X_Train)
            decision_test = perceptron_model.decision_function(self.data_loader.X_Test)
            decision_values_perc_train.append(decision_train)
            decision_values_perc_test.append(decision_test)

        decision_values_perc_train = np.vstack(decision_values_perc_train).T
        decision_values_perc_test = np.vstack(decision_values_perc_test).T

        self.Y_Pred_Perc_Train = self.classes[np.argmax(decision_values_perc_train, axis=1)]
        self.Y_Pred_Perc_Test = self.classes[np.argmax(decision_values_perc_test, axis=1)]

    def compute_metrics(self):
        # Compute performance metrics for aggregated results
        self.metrics = {}
        for model_name, Y_Pred_Train, Y_Pred_Test in [
            ('SVM', self.Y_Pred_SVM_Train, self.Y_Pred_SVM_Test),
            ('Perceptron', self.Y_Pred_Perc_Train, self.Y_Pred_Perc_Test)
        ]:
            self.metrics[model_name] = {
                'Accuracy_Train': accuracy_score(self.data_loader.Y_Train, Y_Pred_Train),
                'Accuracy_Test': accuracy_score(self.data_loader.Y_Test, Y_Pred_Test),
                'Confusion_Matrix_Train': confusion_matrix(self.data_loader.Y_Train, Y_Pred_Train),
                'Confusion_Matrix_Test': confusion_matrix(self.data_loader.Y_Test, Y_Pred_Test)
            }

    def visualize_decision_boundaries(self):
        # Visualization of decision boundaries for aggregated models
        colors = ['blue', 'red', 'orange']
        cmap = ListedColormap(colors)
        markers_train = ['o', 's', '^']
        markers_test = ['*', '*', '*']

        X_Train = self.data_loader.X_Train
        Y_Train = self.data_loader.Y_Train
        X_Test = self.data_loader.X_Test
        Y_Test = self.data_loader.Y_Test

        x_min, x_max = X_Train[:, 0].min() - 1, X_Train[:, 0].max() + 1
        y_min, y_max = X_Train[:, 1].min() - 1, X_Train[:, 1].max() + 1
        h = .02
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        models = {
            'SVM': self.svm_models,
            'Perceptron': self.perceptron_models
        }

        for model_name, model_dict in models.items():
            # Predict over meshgrid
            decision_values = []
            for class_label in self.classes:
                model = model_dict[class_label]
                decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
                decision_values.append(decision)
            decision_values = np.vstack(decision_values).T
            Z = self.classes[np.argmax(decision_values, axis=1)]
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(10, 6))
            plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

            # Plot training points
            for idx, cl in enumerate(self.classes):
                plt.scatter(
                    X_Train[Y_Train == cl, 0],
                    X_Train[Y_Train == cl, 1],
                    c=colors[idx], marker=markers_train[idx],
                    label=f'Train Class {cl}', edgecolor='k', s=100
                )

            # Plot test points
            for idx, cl in enumerate(self.classes):
                plt.scatter(
                    X_Test[Y_Test == cl, 0],
                    X_Test[Y_Test == cl, 1],
                    c=colors[idx], marker=markers_test[idx],
                    label=f'Test Class {cl}', edgecolor='k', s=200
                )

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Decision Surface of Aggregated {model_name} (One-vs-Rest)')
            plt.legend()
            plt.show()

    @classmethod
    def showcase(cls):
        # Instantiate the DataLoader and load data
        data_loader = DataLoader()
        data_loader.load_data()

        # Instantiate the OneVsRestClassifierPart3 with the DataLoader
        ovr_classifier = cls(data_loader)

        # Train models
        ovr_classifier.train_models()

        # Predict
        ovr_classifier.predict()

        # Compute metrics
        ovr_classifier.compute_metrics()

        # Display metrics
        for model_name in ['SVM', 'Perceptron']:
            print(f"\nAggregated {model_name} Metrics:")
            metrics = ovr_classifier.metrics[model_name]
            print(f"Accuracy (Train): {metrics['Accuracy_Train']:.2f}")
            print(f"Accuracy (Test): {metrics['Accuracy_Test']:.2f}")
            print("Confusion Matrix (Train):")
            print(metrics['Confusion_Matrix_Train'])
            print("Confusion Matrix (Test):")
            print(metrics['Confusion_Matrix_Test'])

        # Visualize decision boundaries
        ovr_classifier.visualize_decision_boundaries()


# Part 4: Hyperparameter Tuning for SVM
class SVMHyperparameterTuner(SVMClassifier):
    def tune_hyperparameters(self, C_values):
        # Perform hyperparameter tuning over a range of C values
        param_grid = {'C': C_values}
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=3)
        grid_search.fit(self.data_loader.X_Train, self.data_loader.Y_Train)

        # Best parameter
        self.best_C = grid_search.best_params_['C']
        print(f"Best C parameter: {self.best_C}")

        # Update the classifier with best C
        self.clf = svm.SVC(C=self.best_C)
        # Retrain with the best parameter using inherited train() method
        self.train()

    @classmethod
    def showcase(cls):
        # Instantiate the DataLoader and load data
        data_loader = DataLoader()
        data_loader.load_data()

        # Instantiate the SVMHyperparameterTuner with the DataLoader
        tuner = cls(data_loader)

        # Define a range of C values for tuning
        C_values = np.arange(0.001, 100, 0.1)

        # Perform hyperparameter tuning
        tuner.tune_hyperparameters(C_values)

        # Predict on training and test data
        tuner.predict()

        # Get confusion matrices
        cm_train, cm_test = tuner.get_confusion_matrices()

        # Display confusion matrices
        print("Confusion Matrix for Training Data after Hyperparameter Tuning:")
        print(cm_train)
        print("\nConfusion Matrix for Test Data after Hyperparameter Tuning:")
        print(cm_test)

        # Visualize decision surfaces
        tuner.visualize_decision_surface()

        # Plot validation curve
        train_scores, test_scores = validation_curve(
            svm.SVC(),
            data_loader.X_Train,
            data_loader.Y_Train,
            param_name="C",
            param_range=C_values,
            cv=3,
            scoring='accuracy'
        )

        # Calculate mean and standard deviation
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # Plot validation curve
        plt.figure(figsize=(8, 6))
        plt.plot(C_values, train_scores_mean, label='Training score', marker='o')
        plt.plot(C_values, test_scores_mean, label='Cross-validation score', marker='s')
        plt.xlabel('C parameter')
        plt.ylabel('Accuracy')
        plt.title('Validation Curve for SVM')
        plt.legend()
        plt.xscale('log')
        plt.show()


# ===============================================================================================================
# K-Nearest Neighbors (KNN) Classifier on Car Evaluation Dataset

# Part 1: Data Preparation
class DataPreparation:
    def __init__(self, filepath):
        self.data_frame = pd.read_csv(filepath)
        self.data_frame.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    def shuffle_data(self, random_state=23):
        # Shuffle the dataset
        self.data_frame = shuffle(self.data_frame, random_state=random_state)

    def split_data(self, train_size, validation_size, test_size):
        # Split the dataset into training, validation, and testing sets
        train_dataset = self.data_frame.iloc[:train_size].reset_index(drop=True)
        validation_dataset = self.data_frame.iloc[train_size:train_size + validation_size].reset_index(drop=True)
        test_dataset = self.data_frame.iloc[
                       train_size + validation_size:train_size + validation_size + test_size].reset_index(drop=True)
        return train_dataset, validation_dataset, test_dataset


# Part 2: Data Preprocessing
class DataPreprocessing:
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data.copy()
        self.validation_data = validation_data.copy()
        self.test_data = test_data.copy()
        self.label_encoder = LabelEncoder()

    def encode_features(self):
        # Combine all data for consistent encoding
        combined_data = pd.concat([self.train_data, self.validation_data, self.test_data],
                                  keys=['train', 'validation', 'test'])

        # Separate features and labels
        X = combined_data.drop('class', axis=1)
        Y = combined_data['class']

        # One-Hot Encode features
        X_encoded = pd.get_dummies(X)
        # Label Encode target
        Y_encoded = self.label_encoder.fit_transform(Y)

        # Split back into train, validation, and test sets
        X_train = X_encoded.xs('train')
        X_validation = X_encoded.xs('validation')
        X_test = X_encoded.xs('test')

        Y_train = Y_encoded[:len(X_train)]
        Y_validation = Y_encoded[len(X_train):len(X_train) + len(X_validation)]
        Y_test = Y_encoded[len(X_train) + len(X_validation):]

        return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


# Part 3: Impact of Training Set Size on KNN Performance
class TrainingSizeAnalysis:
    def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
        self.X_train_full = X_train
        self.Y_train_full = Y_train

        self.X_validation = X_validation
        self.Y_validation = Y_validation

        self.X_test = X_test
        self.Y_test = Y_test

        # Percentages from 10% to 100%
        self.training_sizes = range(10, 110, 10)
        self.validation_accuracies = []
        self.testing_accuracies = []

    def train_and_evaluate(self, k=2):
        total_samples = len(self.X_train_full)
        for size in self.training_sizes:
            sample_size = int((size / 100) * total_samples)
            X_train_subset = self.X_train_full[:sample_size]
            Y_train_subset = self.Y_train_full[:sample_size]

            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_subset, Y_train_subset)

            # Evaluate on validation set
            y_val_pred = knn.predict(self.X_validation)
            val_accuracy = accuracy_score(self.Y_validation, y_val_pred)
            self.validation_accuracies.append(val_accuracy)

            # Evaluate on testing set
            Y_test_pred = knn.predict(self.X_test)
            test_accuracy = accuracy_score(self.Y_test, Y_test_pred)
            self.testing_accuracies.append(test_accuracy)
            print(
                f"Training size: {size}% ({sample_size} samples), Validation Accuracy: {val_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_sizes, self.validation_accuracies, marker='o', label='Validation Accuracy')
        plt.plot(self.training_sizes, self.testing_accuracies, marker='s', label='Testing Accuracy')
        plt.title('Accuracy vs. Training Set Size')
        plt.xlabel('Training Set Size (%)')
        plt.ylabel('Accuracy Score')
        plt.xticks(self.training_sizes)
        plt.legend()
        plt.grid(True)
        plt.show()


# Part 4: Tuning the K Parameter
class KParameterTuning:
    def __init__(self, X_train, Y_train, X_validation, Y_validation):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.k_values = range(1, 11)
        self.validation_accuracies = []

    def tune_k(self):
        for k in self.k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.Y_train)
            y_val_pred = knn.predict(self.X_validation)
            val_accuracy = accuracy_score(self.Y_validation, y_val_pred)
            self.validation_accuracies.append(val_accuracy)
            print(f"K={k}, Validation Accuracy: {val_accuracy:.4f}")

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.k_values, self.validation_accuracies, marker='o')
        plt.title('Validation Accuracy vs. K Value')
        plt.xlabel('K Value')
        plt.ylabel('Validation Accuracy')
        plt.xticks(self.k_values)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Exploring SVM and Perceptron Performance (SVM vs Perceptron)

    # Part 1: Default SVM Implementation and Analysis
    # print("========== Part 1: Default SVM Implementation and Analysis ==========\n")
    # SVMClassifier.showcase()

    # Part 2: One-vs-Rest SVM and Perceptron Analysis
    # print("========== Part 2: One-vs-Rest SVM and Perceptron Analysis ==========\n")
    # OneVsRestClassifierPart2.showcase()

    # print("========== Part 2: One-vs-Rest With Noise SVM and Perceptron Analysis ==========\n")
    # OneVsRestClassifierPart2WithNoise.showcase()

    # Part 3: Aggregated Results from One-vs-Rest Strategy
    # print("========== Part 3: Aggregated Results from One-vs-Rest Strategy ==========\n")
    # OneVsRestClassifierPart3.showcase()

    # Part 4: Hyperparameter Tuning for SVM
    # print("========== Part 4: Hyperparameter Tuning for SVM ==========\n")
    # SVMHyperparameterTuner.showcase()

    # ================================================================================================
    # K-Nearest Neighbors (KNN) Classifier on Car Evaluation Dataset

    # Part 1: Data Preparation
    print("\n--- Part 1: Data Preparation ---")
    data_prep = DataPreparation(filepath="data/car_evaluation.csv")
    data_prep.shuffle_data()
    train_data, validation_data, test_data = data_prep.split_data(train_size=1000, validation_size=300, test_size=428)

    # Part 2: Data Preprocessing
    print("\n--- Part 2: Data Preprocessing ---")
    preprocessor = DataPreprocessing(train_data, validation_data, test_data)
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = preprocessor.encode_features()

    # Part 3: Impact of Training Set Size on KNN Performance
    print("\n--- Part 3: Impact of Training Set Size on KNN Performance ---")
    training_size_analysis = TrainingSizeAnalysis(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
    training_size_analysis.train_and_evaluate(k=2)
    training_size_analysis.plot_results()

    # Part 4: Tuning the K Parameter
    print("\n--- Part 4: Tuning the K Parameter ---")
    k_tuner = KParameterTuning(X_train, Y_train, X_validation, Y_validation)
    k_tuner.tune_k()
    k_tuner.plot_results()

    # Part 5: Conclusion and Analysis
    # You can write your analysis based on the outputs from Part 3 and Part 4.
