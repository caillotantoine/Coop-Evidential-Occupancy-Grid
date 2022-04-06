#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:14:15 2022

@author: caillot
"""

from curses import window
from curses.textpad import Textbox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import threading

import time

class GUI:
    mutex:threading.Lock
    paused:bool
    quit:bool
    test:int

    def __init__(self) -> None:
        self.mutex = threading.Lock()
        self.paused = False
        self.quitted = False
        self.test = 0

    def toggle_play_pause(self, event):
        print('play pause')
        self.mutex.acquire()
        self.paused = not self.paused
        self.mutex.release()


    def quit(self, event):
        print('quit')
        self.mutex.acquire()
        self.quitted = True
        self.mutex.release()

    def is_paused(self):
        self.mutex.acquire()
        res = self.paused
        self.mutex.release()
        return res
    
    def is_quitted(self):
        self.mutex.acquire()
        res = self.quitted
        self.mutex.release()
        return res

def gui_thread(state:GUI):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    fig.canvas.mpl_connect('close_event', state.quit)
    # axtxt = plt.axes([0.2, 0.1, 0.6, 0.05])
    axpause = plt.axes([0.7, 0.05, 0.1, 0.075])
    axquit = plt.axes([0.81, 0.05, 0.1, 0.075])
    bpause = Button(axpause, 'Pause')
    bpause.on_clicked(state.toggle_play_pause)
    bquit = Button(axquit, 'Quit')
    bquit.on_clicked(state.quit)
    btxt = plt.text(-5, 0.8, f'Frame {state.test}')
    while True:
        state.test += 1
        btxt.set_text(f"Frame {state.test}")
        plt.pause(0.01)
        if state.is_quitted():
            return

def main():
    state = GUI()
    gui_tread = threading.Thread(target=gui_thread, args=(state,), daemon=True)
    gui_tread.start()
    for i in range(300):
        print(f"Traitement de la frame {i} : {state.test}")
        time.sleep(3)
        while state.is_paused():
            time.sleep(0.2)
            if state.is_quitted():
                exit()
        if state.is_quitted():
            gui_tread.join()
            plt.close()
            exit()
        


if __name__ == "__main__":
    main()
