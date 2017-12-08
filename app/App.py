# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:21:17 2017

@author: Chens
"""

import tkinter as tk
from tkinter import ttk

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style


LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

f = Figure(figsize=(5,5), dpi=100)
a = f.add_subplot(111)


def animate(i):
    pullData = open("SampleData.txt", "r").read()
    dataList = pullData.split('\n')
    xList = []
    yList = []
    for eachline in dataList:
        if len(eachline) > 1:
            x, y = eachline.split(',')
            xList.append(int(x))
            yList.append(int(y))
    a.clear()
    a.plot(xList, yList)
    

    
"""
The main App: 
"""
class StockApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        tk.Tk.iconbitmap(self, 'example.ico')
        tk.Tk.wm_title(self, "Stock Analyzer App")
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        # Container for all different frames.. e.g. pages
        self.frames = {}
        
        for F in (StartPage, PageOne, PageThree):
            frame = F(container, self)
            self.frames[F] = frame
            
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        
        frame = self.frames[cont]
        # raise the fram to the front
        frame.tkraise()

"""
The start Page: 
"""        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # add text to window 
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10) # adding paddings around to look neat
        # Define Button here
        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button_graph = ttk.Button(self, text="Go to Graph Page",
                            command=lambda: controller.show_frame(PageThree))
        button_graph.pack()

class PageOne(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # add text to window 
        label = tk.Label(self, text="Page One", font=LARGE_FONT)
        label.pack(pady=10,padx=10) # adding paddings around to look neat        
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()    

class PageThree(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # add text to window 
        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10) # adding paddings around to look neat        
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()   
        # plot data in background
        #f = Figure(figsize=(5,5), dpi=100)
        #a = f.add_subplot(111)
        #a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
        
        #make plt.show() show up on canvas 
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        



def qf(quickPrint):
    print(quickPrint)


app = StockApp()
ani = animation.FuncAnimation(f, animate, interval=1000)
app.mainloop()
        
