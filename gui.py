import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
#from FinalProject import process_data  # Assuming there's a processing function called process_data in 'FinalProject.py'


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)


def process_and_label_data():
    file_path = file_entry.get()
    if not file_path:
        messagebox.showerror("Error", "Please select a CSV file.")
        return

    try:
        # Process the data and get labeled output
        #labeled_data = process_data(file_path) THIS WOULD CALL THE FUNCTION IN PART 6

        # Output labels to a new CSV file
        output_file_path = file_path.rsplit('.', 1)[0] + "_labeled.csv"
        #labeled_data.to_csv(output_file_path, index=False)

        messagebox.showinfo("Success", f"Data labeled and saved to:\n{output_file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Error processing data: {e}")


# GUI setup
root = tk.Tk()
root.title("Data Labeling")

file_label = tk.Label(root, text="Select CSV file:")
file_label.grid(row=0, column=0, padx=10, pady=10)

file_entry = tk.Entry(root, width=40)
file_entry.grid(row=0, column=1, padx=10, pady=10)

browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, padx=10, pady=10)

process_button = tk.Button(root, text="Process & Label Data", command=process_and_label_data)
process_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()