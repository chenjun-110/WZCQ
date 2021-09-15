import win32gui
hwnd_title = dict()
def get_all_hwnd(hwnd,mouse):
  if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
    hwnd_title.update({hwnd:win32gui.GetWindowText(hwnd)})
 
win32gui.EnumWindows(get_all_hwnd, 0)
titles = []
for h,t in hwnd_title.items():
  if t is not "":
    titles.append(t)
    print('h:',h, ' t:',t)

import win32gui
from PIL import ImageGrab
import win32con

hwnd_ = win32gui.FindWindow(None,"doNotTouchWhite - Google Chrome")
if not hwnd_:
    print( 'window not found!')
else:
    print('hwnd_',hwnd_)

# win32gui.ShowWindow(hwnd_, win32con.SW_RESTORE) # 强行显示界面后才好截图
win32gui.SetForegroundWindow(hwnd_)  # 将窗口提到最前
game_rect = win32gui.GetWindowRect(hwnd_)
src_image = ImageGrab.grab(game_rect) #截桌面
# src_image = ImageGrab.grab((game_rect[0] + 9, game_rect[1] + 190, game_rect[2] - 9, game_rect[1] + 190 + 450))
src_image.show()


# #对后台窗口截图
# import win32gui, win32ui, win32con
# from ctypes import windll
# from PIL import Image
# import cv2
# import numpy
# #获取后台窗口的句柄，注意后台窗口不能最小化
# # hWnd = win32gui.FindWindow(None, titles[5]) #窗口的类名可以用Visual Studio的SPY++工具获取
# hWnd = win32gui.FindWindow('Chrome_WidgetWin_1', None) #窗口的类名可以用Visual Studio的SPY++工具获取

# #获取句柄窗口的大小信息
# left, top, right, bot = win32gui.GetWindowRect(hWnd)
# width = right - left
# height = bot - top
# #返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
# hWndDC = win32gui.GetWindowDC(hWnd)
# #创建设备描述表
# mfcDC = win32ui.CreateDCFromHandle(hWndDC)
# #创建内存设备描述表
# saveDC = mfcDC.CreateCompatibleDC()
# #创建位图对象准备保存图片
# saveBitMap = win32ui.CreateBitmap()
# #为bitmap开辟存储空间
# saveBitMap.CreateCompatibleBitmap(mfcDC,width,height)
# #将截图保存到saveBitMap中
# saveDC.SelectObject(saveBitMap)
# #保存bitmap到内存设备描述表
# saveDC.BitBlt((0,0), (width,height), mfcDC, (0, 0), win32con.SRCCOPY)
# ##方法三
# signedIntsArray = saveBitMap.GetBitmapBits(True)
# im_opencv = numpy.frombuffer(signedIntsArray, dtype = 'uint8')
# im_opencv.shape = (height, width, 4)
# cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)
# cv2.imwrite("im_opencv.jpg",im_opencv,[int(cv2.IMWRITE_JPEG_QUALITY), 100]) #保存
# cv2.namedWindow('im_opencv') #命名窗口
# cv2.imshow("im_opencv",im_opencv) #显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #如果要截图到打印设备：
# ###最后一个int参数：0-保存整个窗口，1-只保存客户区。如果PrintWindow成功函数返回值为1
# result = windll.user32.PrintWindow(hWnd,saveDC.GetSafeHdc(),0)
# print(result) #PrintWindow成功则输出1
 
# #保存图像
# ##方法一：windows api保存
# ###保存bitmap到文件
# saveBitMap.SaveBitmapFile(saveDC,"img_Winapi.bmp")
 
##方法二(第一部分)：PIL保存
###获取位图信息
# bmpinfo = saveBitMap.GetInfo()
# bmpstr = saveBitMap.GetBitmapBits(True)
# ###生成图像
# im_PIL = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr,'raw','BGRX',0,1)
##方法二（后续转第二部分）
 
##方法三（第一部分）：opencv+numpy保存
###获取位图信息
##方法三（后续转第二部分）
 
#内存释放
win32gui.DeleteObject(saveBitMap.GetHandle())
saveDC.DeleteDC()
mfcDC.DeleteDC()
win32gui.ReleaseDC(hWnd,hWndDC)
 
##方法二（第二部分）：PIL保存
###PrintWindow成功,保存到文件,显示到屏幕
# im_PIL.save("im_PIL.png") #保存
# im_PIL.show() #显示
 
##方法三（第二部分）：opencv+numpy保存
###PrintWindow成功，保存到文件，显示到屏幕



