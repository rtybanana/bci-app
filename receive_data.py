import threading
from sys import path
from pylsl import StreamInlet, resolve_stream
from queue import Queue
import numpy as np

path.append("EEGModels")
from evaluation import EEGNet


trial_queue = Queue()

def classify(done, model):
  while not done() or not trial_queue.empty():
    if not trial_queue.empty():
        trial = trial_queue.get()
        probs = model.predict(trial)
        print(f"classified {probs.argmax()}")
        # socket.sendall(bytes([probs.argmax()]))
      

def stream(model):
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    done = False
    x = threading.Thread(target=classify, args=(lambda: done, model))
    x.start()

    window = Queue()

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        try:
            sample, timestamp = inlet.pull_sample()
            window.put(sample)
            if (window.qsize() > 176):
                window.get()
                trial = np.array([[list(window.queue)]])
                trial = trial.reshape(1, 1, 25, 176)
                # print(trial.shape)

                trial_queue.put(trial)
        except KeyboardInterrupt:
            print("cancel thread")
            break

    done = True
    x.join()

def main():
    model = EEGNet(nb_classes=2, Chans=25, Samples=176, dropoutRate=0.5, 
                    kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout')

    model.load_weights(f'BCICIV2a/weights/all.h5')
    model._make_predict_function() 
    stream(model)

main()