import numpy as np
import matplotlib.pyplot as plt
import importlib

# Import the 'SBC-Analysis' module
sbc = importlib.import_module('SBC-Analysis')

# Choose an event file by its index (i)
i = 1

this_event = sbc.get_event('/Users/danielcampos/python/DAHL CODE/Training_data/20250307_0/', i, 'fastDAQ')

voltages = this_event['fastDAQ']['Piez_top']
time = this_event['fastDAQ']['time']

# Determine the sampling interval (assumes uniform sampling)
dt = time[1] - time[0]
# Calculate the number of samples in a 20ms chunk
chunk_duration = 0.02  # 20 milliseconds in seconds
samples_per_chunk = int(chunk_duration / dt)

# Calculate how many full chunks are available in the event
max_chunk_index = len(time) // samples_per_chunk

# Global variable to track which chunk we're displaying (starting at 0)
chunk_index = 0

def plot_chunk(chunk_idx):
    """Plot a 20ms chunk of the event signal based on the chunk index."""
    start = chunk_idx * samples_per_chunk
    end = start + samples_per_chunk
    if end > len(time):
        end = len(time)
    plt.clf()  # Clear the previous plot
    plt.plot(time[start:end], voltages[start:end], label=f'Chunk {chunk_idx+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    plt.ylim(-0.6,0.6)
    plt.title(f'Event {i} | Chunk {chunk_idx+1} of {max_chunk_index}')
    plt.legend()
    plt.draw()  # Redraw the figure

def on_key(event):
    """Handle key press events to navigate through chunks."""
    global chunk_index
    if event.key == 'right':  # Forward: next chunk
        if chunk_index < max_chunk_index - 1:
            chunk_index += 1
            plot_chunk(chunk_index)
        else:
            print("Reached the last chunk.")
    elif event.key == 'left':  # Backward: previous chunk
        if chunk_index > 0:
            chunk_index -= 1
            plot_chunk(chunk_index)
        else:
            print("Reached the first chunk.")

# Set up the interactive plot
fig = plt.figure()
plot_chunk(chunk_index)  # Plot the first 20ms chunk

# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
