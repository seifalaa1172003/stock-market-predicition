import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from Main_Models import get_models, get_preprocessing_components
import numpy as np

class StockPricePredictorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stock Price Predictor")

        # Labels and entry fields for input
        self.labels = ["AMZN:", "DPZ:", "BTC:", "NFLX:"]
        self.entries = []
        for i, label_text in enumerate(self.labels):
            label = tk.Label(master, text=label_text)
            label.grid(row=i, column=0, sticky="e")
            entry = tk.Entry(master)
            entry.grid(row=i, column=1)
            self.entries.append(entry)

        # Dropdown menu for model selection
        self.model_label = tk.Label(master, text="Select Model:")
        self.model_label.grid(row=4, column=0, sticky="e")
        self.model_var = tk.StringVar(master)
        self.model_var.set("Logistic Regression")
        self.model_dropdown = tk.OptionMenu(master, self.model_var, "Logistic Regression", "Random Forest", "Decision Tree", "SVM")
        self.model_dropdown.grid(row=4, column=1)

        # Prediction button
        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=5, columnspan=2)

        # Output label for prediction
        self.prediction_label = tk.Label(master, text="")
        self.prediction_label.grid(row=6, columnspan=2)

        # Buttons for metrics
        self.accuracy_button = tk.Button(master, text="Accuracy", command=self.calculate_accuracy)
        self.accuracy_button.grid(row=7, column=0)
        self.accuracy_output = scrolledtext.ScrolledText(master, width=30, height=1)
        self.accuracy_output.grid(row=7, column=1)

        self.precision_button = tk.Button(master, text="Precision", command=self.calculate_precision)
        self.precision_button.grid(row=8, column=0)
        self.precision_output = scrolledtext.ScrolledText(master, width=30, height=1)
        self.precision_output.grid(row=8, column=1)

        self.cv_button = tk.Button(master, text="CV Scores", command=self.calculate_cv_scores)
        self.cv_button.grid(row=9, column=0)
        self.cv_output = scrolledtext.ScrolledText(master, width=70, height=1)
        self.cv_output.grid(row=9, column=1)

        self.mean_cv_button = tk.Button(master, text="Mean CV Score", command=self.calculate_mean_cv_score)
        self.mean_cv_button.grid(row=10, column=0)
        self.mean_cv_output = scrolledtext.ScrolledText(master, width=40, height=1)
        self.mean_cv_output.grid(row=10, column=1)

        # Tab to display confusion matrix
        self.tab_parent = ttk.Notebook(master)
        self.tab_confusion_matrix = ttk.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_confusion_matrix, text="Confusion Matrix")
        self.tab_parent.grid(row=11, columnspan=2)

        # Text area to display confusion matrix
        self.confusion_matrix_text = tk.Text(self.tab_confusion_matrix, width=80, height=10)
        self.confusion_matrix_text.grid(row=0, column=0)
        self.confusion_matrix_text.config(state='disabled')  # Make it un-editable

        # Radio buttons to select model for confusion matrix
        self.model_radio_var = tk.StringVar(value="Logistic Regression")
        self.model_radio_buttons = []
        for i, model_name in enumerate(["Logistic Regression", "Random Forest", "Decision Tree", "SVM"]):
            rb = tk.Radiobutton(self.tab_confusion_matrix, text=model_name, variable=self.model_radio_var, value=model_name)
            rb.grid(row=i, column=1, sticky="w")
            self.model_radio_buttons.append(rb)

        # Button to show confusion matrix
        self.show_confusion_matrix_button = tk.Button(self.tab_confusion_matrix, text="Show Confusion Matrix", command=self.show_confusion_matrix)
        self.show_confusion_matrix_button.grid(row=4, columnspan=2)

        # Get models and preprocessing components from main models file
        self.models = get_models()
        self.X_train, self.X_test, self.y_train, self.y_test, self.label_encoder, self.scaler = get_preprocessing_components()

    def preprocess_input(self, input_data):
        # Preprocess the input data similar to training data
        # Perform any necessary transformations such as scaling and encoding
        # Return the preprocessed input data
        # For simplicity, assume input_data is a list containing values for AMZN, DPZ, BTC, NFLX
        input_array = np.array([input_data], dtype=np.float64)
        input_scaled = self.scaler.transform(input_array)
        return input_scaled

    def predict(self):
        # Get user input
        input_data = [entry.get() for entry in self.entries]

        # Preprocess input data
        input_scaled = self.preprocess_input(input_data)

        # Get selected model
        selected_model = self.model_var.get()

        # Perform prediction using the selected model
        if selected_model == "Logistic Regression":
            model = self.models[0]
        elif selected_model == "Random Forest":
            model = self.models[1]
        elif selected_model == "Decision Tree":
            model = self.models[2]
        elif selected_model == "SVM":
            model = self.models[3]

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_movement = self.label_encoder.inverse_transform(prediction)[0]  # Convert label to string

        # Display prediction
        self.prediction_label.config(text=f"Prediction: {predicted_movement}")

    # Methods for calculating metrics
    def calculate_accuracy(self):
        selected_model = self.model_var.get()
        model_index = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"].index(selected_model)
        model = self.models[model_index]
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.accuracy_output.config(state='normal')
        self.accuracy_output.delete(1.0, tk.END)  # Clear previous text
        self.accuracy_output.insert(tk.END, f"Accuracy: {accuracy *100:.4f}%")
        self.accuracy_output.config(state='disabled')

    def calculate_precision(self):
        selected_model = self.model_var.get()
        model_index = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"].index(selected_model)
        model = self.models[model_index]
        y_pred = model.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        self.precision_output.config(state='normal')
        self.precision_output.delete(1.0, tk.END)  # Clear previous text
        self.precision_output.insert(tk.END, f"Precision: {precision *100:.4f}%")
        self.precision_output.config(state='disabled')

    def calculate_cv_scores(self):
        selected_model = self.model_var.get()
        model_index = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"].index(selected_model)
        model = self.models[model_index]
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        self.cv_output.config(state='normal')
        self.cv_output.delete(1.0, tk.END)  # Clear previous text
        self.cv_output.insert(tk.END, f"CV Scores: {cv_scores}")
        self.cv_output.config(state='disabled')

    def calculate_mean_cv_score(self):
        selected_model = self.model_var.get()
        model_index = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"].index(selected_model)
        model = self.models[model_index]
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        mean_cv_score = cv_scores.mean()
        self.mean_cv_output.config(state='normal')
        self.mean_cv_output.delete(1.0, tk.END)  # Clear previous text
        self.mean_cv_output.insert(tk.END, f"Mean CV Score: {mean_cv_score}")
        self.mean_cv_output.config(state='disabled')

    def show_confusion_matrix(self):
        selected_model = self.model_radio_var.get()
        model_index = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"].index(selected_model)
        model = self.models[model_index]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        # Format the confusion matrix
        cm_str = "Confusion Matrix:\n\n"
        cm_str += f"\t\tPredicted Up\t\tPredicted Down\t\tPredicted Flat\n"
        cm_str += f"Actual Up\t\t{cm[0,0]}\t\t\t{cm[0,1]}\t\t\t{cm[0,2]}\n"
        cm_str += f"Actual Down\t\t{cm[1,0]}\t\t\t{cm[1,1]}\t\t\t{cm[1,2]}\n"
        cm_str += f"Actual Flat\t\t{cm[2,0]}\t\t\t{cm[2,1]}\t\t\t{cm[2,2]}\n"

        self.confusion_matrix_text.config(state='normal')
        self.confusion_matrix_text.delete(1.0, tk.END)  # Clear previous text
        self.confusion_matrix_text.insert(tk.END, cm_str)
        self.confusion_matrix_text.config(state='disabled')

# Create and run the GUI
root = tk.Tk()
app = StockPricePredictorGUI(root)
root.mainloop()
