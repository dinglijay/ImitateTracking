# from conf.configs import ADNetConf
# import multiprocessing as mp
# import os



# def worker(num):
#     """thread worker function"""
#     print ('Worker:', num)
#     print(ADNetConf.get()['dl_paras']['zoom_scale'])
#     return


# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())

# def f(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     ADNetConf.get('conf/dylan.yaml')
#     # info('main line')
#     # p = mp.Process(target=f, args=('bob',))
#     # p.start()
#     # p.join()
    
#     jobs = []
#     ctx = mp.get_context('spawn')
#     for i in range(5):

#         p = ctx.Process(target=worker, args=(i,))
#         jobs.append(p)
#         p.start()

# # def worker(num, input_queue):
# #     """thread worker function"""
# #     print('Worker:', num)
# #     for inp in iter(input_queue.get, 'stop'):
# #         print('executing %s' % inp)
# #         # os.popen('./mmul.x < '+inp+' >'+inp+'.out' )
# #     return

# # if __name__ == '__main__':
# #     input_queue = mp.Queue() # queue to allow communication
# #     for i in range(4):
# #         input_queue.put('in'+str(i))   # the queue contains the name of the inputs  
# #     for i in range(4):
# #         input_queue.put('stop')  # add a poison pill for each process
# #     for i in range(4):
# #         p = mp.Process(target=worker, args=(i, input_queue))
# #         p.start()


from PIL import Image

@profile
def func1():
    with Image.open('0028.jpg') as test_image:
        print('ok')

@profile
def func2():
    ob =  Image.open('0028.jpg') 
    print('ok')


for i in range(1000):
    func1()
    func2()