from ctypes import cdll
from os import system

system("make")
lib = cdll.LoadLibrary('./libmain.so')