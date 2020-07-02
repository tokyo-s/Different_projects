#Plan:
#5.Dice animation
#8.Choosing colors
import pygame
import tkinter as tk
from tkinter import messagebox
import random
import os

class Player:
    def __init__(self,x,y,color):
        self.pos =1
        self.x=454
        self.y=454
        self.color=color
        self.turns=0

    def Turn(self,die1,die2):
        self.turns+=1
        self.pos+=die1+die2
        if self.pos==100: 
            return
        if self.pos>100: self.pos=100-(self.pos%100)
        if self.pos==2: self.pos=38
        elif self.pos==7: self.pos=14
        elif self.pos==8: self.pos=31
        elif self.pos==15: self.pos=26
        elif self.pos==16: self.pos=6
        elif self.pos==21: self.pos=42
        elif self.pos==28: self.pos=84
        elif self.pos==36: self.pos=44
        elif self.pos==46: self.pos=25
        elif self.pos==49: self.pos=11
        elif self.pos==51: self.pos=67
        elif self.pos==62: self.pos=19
        elif self.pos==64: self.pos=60
        elif self.pos==71: self.pos=91
        elif self.pos==74: self.pos=53
        elif self.pos==78: self.pos=98
        elif self.pos==87: self.pos=94
        elif self.pos==89: self.pos=68
        elif self.pos==92: self.pos=88
        elif self.pos==95: self.pos=75
        elif self.pos==99: self.pos=80

    def draw(self):
        global red,blue#need it?
        if self.color=='green':
            screen.blit(green,(0,0))#change position to one relative to pos from player
        elif self.color=='blue':
            screen.blit(blue,(0,0))
        elif self.color=='red':
            screen.blit(red,(0,0))
        elif self.color=='yellow':
            screen.blit(yellow,(0,0))     #i should draw second player anyway
    #def Move():


def redrawWindow(surface,player1,player2):
    global height,background
    screen.blit(background,(0,0))
    player1.draw()
    player2.draw()
    pygame.display.update()

def messageBox(subject,content):
    root=tk.Tk()
    root.attributes("-topmost",True)
    root.withdraw()
    messagebox.showinfo(subject,content)
    try:
        root.destroy()
    except:
        pass

def main():
    global height,red,blue,background
    height =736
    win=pygame.display.set_mode((height,height))
    flag=True
    clock=pygame.time.Clock()
    background = pygame.image.load(os.path.join('background.png')).convert()#ye ye images, now u fucked up 
    red= pygame.image.load('red.png')
    blue= pygame.image.load('blue.png')
    p1=Player(x,y,red)#yes and about coordinates i dont give a shit so its your turn
    p2=Player(x,y,blue)
    players=[p1,p2]
    while flag:
        pygame.time.delay(50)
        screen.blit(background,(0,0))
        clock.tick(2)       ###################here change if unconventionla
        die1=random.randrange(1,7)
        die2=random.randrange(1,7)
        players[0].Turn(die1,die2)
        redrawWindow(win,players[0],players[1])
        if players[0].pos==100:messageBox('You won!',"In {players[0].turns} turns")#####################
        if die1==die2:continue
        else: players.reverse()
        
main()