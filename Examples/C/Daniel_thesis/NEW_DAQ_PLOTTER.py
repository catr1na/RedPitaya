#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:35:07 2024

@author: danielcampos
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatches

# Import the 'SBC-Analysis' module
sbc = importlib.import_module('SBC-Analysis')

# Initial index value
i = 22

# Global references
ax = None
rect_selector = None

def update_rect_props():
    """Attempt to update the rectangle patch properties to remove fill."""
    global rect_selector, ax
    rect_patch = None
    # Try different attribute names
    if hasattr(rect_selector, '_rect'):
        rect_patch = rect_selector._rect
    elif hasattr(rect_selector, 'rectangle'):
        rect_patch = rect_selector.rectangle
    else:
        # Fallback: iterate over the axes children to find a Rectangle
        for child in ax.get_children():
            if isinstance(child, mpatches.Rectangle):
                rect_patch = child
                break
    if rect_patch is not None:
        rect_patch.set_facecolor('none')      # Remove fill
        rect_patch.set_edgecolor('blue')        # Set edge color to blue
        rect_patch.set_linestyle('-')           # Solid line
        rect_patch.set_linewidth(1)             # Line width of 1
    else:
        print("Warning: Rectangle patch not found.")

# Function to plot the event data without the transition finder line
def plot_event(i):
    global sbc, ax, rect_selector
    this_event = sbc.get_event('/Users/danielcampos/python/DAHL CODE/Training_data/20250307_0/', i, 'fastDAQ')
    voltages = this_event['fastDAQ']['Piez_top']
    time = this_event['fastDAQ']['time']
    
    # Clear the previous plot and get the current axes
    plt.clf()
    ax = plt.gca()
    ax.plot(time, voltages, label='Signal')
    ax.set_title("DAQ Bubble Event")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    
    # Reset axis limits
    ax.autoscale()
    
    # Initialize the RectangleSelector without using rectprops
    rect_selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1],  # left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    
    # Update rectangle properties to remove the fill color
    update_rect_props()
    
    plt.draw()

# Callback function for key press events to navigate through events
def on_key(event):
    global i
    if event.key == 'right':  # Right arrow key increases i
        if i < 100:
            i += 1
            plot_event(i)
        else:
            print("Reached the maximum i value of 100.")
    elif event.key == 'left':  # Left arrow key decreases i
        if i > 1:
            i -= 1
            plot_event(i)
        else:
            print("Reached the minimum i value of 1.")
    elif event.key == 'escape':  # Reset zoom on escape key
        ax.autoscale()
        plt.draw()

# Callback function for RectangleSelector
def on_select(eclick, erelease):
    # Retrieve the starting and ending coordinates
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    if None not in (x1, y1, x2, y2):
        ax.set_xlim(min(x1, x2), max(x1, x2))
        ax.set_ylim(min(y1, y2), max(y1, y2))
        plt.draw()

# Set up the interactive plot
fig = plt.figure()
plot_event(i)  # Plot the event for the initial index value

# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
