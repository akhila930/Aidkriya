import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime
import os

class AidkriyaPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("aidKRIYA Donation Prediction System")
        self.root.geometry("600x400")
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="aidKRIYA Donation Prediction", 
                              font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Date selection
        date_label = ttk.Label(main_frame, text="Select Date:")
        date_label.grid(row=1, column=0, pady=10, padx=5, sticky=tk.W)
        
        self.date_picker = DateEntry(main_frame, width=20, background='darkblue',
                                   foreground='white', borderwidth=2)
        self.date_picker.grid(row=1, column=1, pady=10, padx=5, sticky=tk.W)
        
        # Predict button
        predict_btn = ttk.Button(main_frame, text="Predict Donation", command=self.predict)
        predict_btn.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.result_label = ttk.Label(result_frame, text="", wraplength=500)
        self.result_label.grid(row=0, column=0, pady=10)
        
        # Load and prepare the model
        self.load_model()

    def load_model(self):
        try:
            # Get the absolute path of the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, "Year-important-dates-data(Sheet1) (1).csv")
            
            # Load dataset
            df = pd.read_csv(file_path)
            
            # Clean and prepare data
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df = df.set_index('Date')
            
            # Use 'Total spend happened(Reserach)' as target
            target_column = 'Total spend happened(Reserach)'
            df[target_column] = pd.to_numeric(df[target_column].str.replace(',', ''), errors='coerce')
            df = df.dropna(subset=[target_column])
            
            # Normalize the values
            self.max_value = df[target_column].max()
            df[target_column] = df[target_column] / self.max_value
            
            # Train SARIMA model
            self.train_data = df[target_column]
            model = SARIMAX(self.train_data, 
                          order=(1, 1, 1), 
                          seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            self.results = model.fit(disp=False)
            
            # Prepare forecast
            self.update_forecast()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")

    def update_forecast(self):
        # Generate forecast for next  days 365
        forecast_steps = 365
        forecast = self.results.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=self.train_data.index[-1], 
                                     periods=forecast_steps + 1, 
                                     freq='D')[1:]
        forecast_values = forecast.predicted_mean
        self.forecast = pd.Series(forecast_values.values, index=forecast_index)

    def predict(self):
        try:
            # Get selected date
            selected_date = pd.to_datetime(self.date_picker.get_date())
            
            # Find the closest date in the forecast index
            if selected_date in self.forecast.index:
                aligned_date = selected_date
            else:
                aligned_date = self.forecast.index[
                    self.forecast.index.get_indexer([selected_date], method='nearest')[0]
                ]
            
            # Get prediction and convert back to actual value
            prediction = self.forecast.loc[aligned_date] * self.max_value
            
            # Format result message
            result_text = (
                f"Predicted Donation for {aligned_date.strftime('%d-%m-%Y')}:\n"
                f"₹{prediction:,.2f}\n\n"
                f"This prediction is based on historical patterns and seasonal trends.\n"
                f"Factors considered include festivals, events, and past donation behavior."
            )
            
            # Update result label
            self.result_label.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = AidkriyaPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
