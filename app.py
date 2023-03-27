import tkinter as tk
from tkinter import *
from chat import get_response, bot_name

#colors & fonts
BG_GRAY = "#17202A"
BG_COLOR = "#0B1214"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApp:
    def __init__(self):
        self.window=Tk()
        self._setup_main_window()

    def _setup_main_window(self):
        self.window.title("ANICKA")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=550, height=630, bg=BG_COLOR)

        #head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                            text='ANICKA 1.0', font=FONT_BOLD, pady=20)
        head_label.place(relwidth=1, relheight=0.075)

        #tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.188, relheight=0.012)

        #text widget
        self.text_widget= Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                            font=FONT_BOLD, padx=5, pady=5)
        self.text_widget.place(relheight=0.74, relwidth=1, rely=0.075)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        #scroll bar
        scroollbar = Scrollbar(self.text_widget)
        scroollbar.place(relheight=1, relx=0.974)
        scroollbar.configure(command=self.text_widget.yview)

        #bottom label
        buttom_label = Label(self.window, bg=BG_GRAY, height=80)
        buttom_label.place(relwidt=1, rely=0.825)

        #message entry box
        self.msg_entry=Entry(buttom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        #send button
        send_button = Button(buttom_label, text="Send", font=FONT_BOLD, width=20, 
                                bg=BG_GRAY, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{bot_name}: {get_response(msg)}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ChatApp()
    app.run()