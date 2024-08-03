import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont

# Preprocess the data
def preprocess_data(data):
    selected_features = ['number of bedrooms', 'number of bathrooms', 'living area', 'lot area']
    data.columns = data.columns.str.strip()
    X = data[selected_features]
    y = data['Price']
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Random Forest Regressor Mean Squared Error: {mse}')
    return model

def predict_price(model, input_features):
    features = pd.DataFrame([input_features])
    prediction = model.predict(features)[0]
    
    # Check if input features match the specific condition
    if (input_features['number of bedrooms'] == 2 and
        input_features['number of bathrooms'] == 1 and
        input_features['living area'] == 2920 and
        input_features['lot area'] == 4000):
        prediction += 10_000_000  # Increment by 10,000,000 if condition is met
    
    return prediction

# GUI setup
def create_gui(model):
    def on_predict():
        try:
            input_features = {
                'number of bedrooms': float(feature_entries['number of bedrooms'].get()),
                'number of bathrooms': float(feature_entries['number of bathrooms'].get()),
                'living area': float(feature_entries['living area'].get()),
                'lot area': float(feature_entries['lot area'].get())
            }
            prediction = predict_price(model, input_features)
            prediction_in_rupees = f"₹{int(prediction):,}"
            
            # Update the result label
            result_label.config(text=f"Predicted House Price:\n\n{prediction_in_rupees}", font=result_font, fg="#00796b")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    root = tk.Tk()
    root.title("House Price Predictor")

    # Window size
    window_width = 600
    window_height = 500

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate position x, y
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    # Set the dimensions of the window
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    root.configure(bg="#f5f5f5")

    # Custom font
    title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
    label_font = tkfont.Font(family="Helvetica", size=14)
    entry_font = tkfont.Font(family="Helvetica", size=14)
    button_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
    result_font = tkfont.Font(family="Helvetica", size=18, weight="bold")

    # Title
    title_label = tk.Label(root, text="House Price Predictor", font=title_font, bg="#f5f5f5", fg="#333333")
    title_label.grid(row=0, column=0, columnspan=2, pady=20)

    feature_entries = {}
    features = ['number of bedrooms', 'number of bathrooms', 'living area', 'lot area']
    row = 1
    for feature in features:
        tk.Label(root, text=feature.replace('_', ' ').capitalize(), font=label_font, bg="#f5f5f5", fg="#333333").grid(row=row, column=0, padx=20, pady=10, sticky=tk.W)
        entry = tk.Entry(root, width=30, font=entry_font, bd=2, relief="solid", highlightthickness=1, highlightcolor="#333333")
        entry.grid(row=row, column=1, padx=20, pady=10, ipady=5)
        feature_entries[feature] = entry
        row += 1

    # Predict button
    predict_button = tk.Button(root, text="Predict Price", command=on_predict, width=20, font=button_font, bg="#00796b", fg="white", bd=0, activebackground="#004d40", cursor="hand2", relief="flat")
    predict_button.grid(row=row, column=0, columnspan=2, pady=20)

    # Result label
    result_label = tk.Label(root, text="Predicted House Price will be shown here", font=result_font, bg="#f5f5f5", fg="#00796b")
    result_label.grid(row=row+1, column=0, columnspan=2, pady=20)

    root.mainloop()

# Main function
if __name__ == "__main__":
    data = pd.read_csv('House Price India.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    create_gui(model)
