## add local modules
import sys, os
from mne.io.pick import pick_channels
sys.path.insert(0, os.path.join(sys.path[0], 'modules'))

# imports
# from evaluation import EEGNet
import numpy as np
from threading import Thread
from time import sleep
from mne import create_info
from mne.io import RawArray
from integration import game_connect, game_disconnect, lsl_connect, lsl_disconnect, get_model, stream_channels, LO_FREQ, HI_FREQ, GOODS
from preparation import filter_channels
import PySimpleGUI as sg
from tkinter.filedialog import askopenfilenames
from collections import deque

#############################################################
### START debug stuff
from preparation import load_pilot, epoch_pilot

CLASSES = 3
FOLDS = 5
REPEATS = 10
GOODS = ['FC3','C3','CP3','Fz','Cz','POz','FC4','C4','CP4']
T_RANGE = [0.5, 2.5]
RESAMPLE = 125
KERNELS = 1
EPOCHS = 200
TRANSFER_EPOCHS = 300
LO_FREQ = 1.
HI_FREQ = 32.
WEIGHT_PATH = f"weights/test"
CONFIDENCE = 0.35

pilot, labels = epoch_pilot(load_pilot('data/rivet/raw/pilot2/BCI_imaginedmoves_3class_7-4-21.vhdr'), CLASSES, GOODS, resample=RESAMPLE, trange=T_RANGE, l_freq=LO_FREQ, h_freq=HI_FREQ)
label_map = ['left', 'right', 'feet']
### END debug stuff
#############################################################

class App:
  # class variables
  layout = [
    [sg.Column(
      [
        [sg.Text("RIVET BCI", font=("Helvetica", 20), pad=((0,0),(0,10)))],
        [sg.Button("Train model", size=(20, 1), key="btn_train_model")],
        [sg.Button("Connect game", size=(20, 1), key="btn_con_game")],
        [sg.Button("Connect headset", size=(20, 1), key="btn_con_headset")],
        [sg.Column(
          [
            [
              sg.Text("Base model loaded", background_color="lightgrey", text_color="black"), 
              sg.Text(u'\u2717', background_color="lightgrey", text_color="red", key="base_model_status", font=("Helvetica", 15)),
              sg.Image('assets/loading.gif', background_color="lightgrey", visible=False, pad=((8,9),(8,8)), key="base_model_loading")
            ],
            [
              sg.Text("Transfer model trained", background_color="lightgrey", text_color="black"), 
              sg.Text(u'\u2717', background_color="lightgrey", text_color="red", key="model_train_status", font=("Helvetica", 15)),
              sg.Image('assets/loading.gif', background_color="lightgrey", visible=False, pad=((8,9),(8,8)), key="model_train_loading")
            ],
            [
              sg.Text("ForestShepherd connected", background_color="lightgrey", text_color="black"), 
              sg.Text(u'\u2717', background_color="lightgrey", text_color="red", key="game_con_status", font=("Helvetica", 15)),
              sg.Image('assets/loading.gif', background_color="lightgrey", visible=False, pad=((8,9),(8,8)), key="game_con_loading")
            ], 
            [
              sg.Text("Headset connected", background_color="lightgrey", text_color="black"), 
              sg.Text(u'\u2717', background_color="lightgrey", text_color="red", key="headset_con_status", font=("Helvetica", 15)),
              sg.Image('assets/loading.gif', background_color="lightgrey", visible=False, pad=((8,9),(8,8)), key="headset_con_loading")
            ]
          ],
          background_color="lightgrey",
          element_justification='left',
          pad=((0,0),(20,20))
        )],
        [sg.Button("Finalize", size=(20, 1), key="btn_finalize")]
      ], 
      element_justification='center'
    )]
  ]

  stream_info = create_info(stream_channels, 500, 'eeg')

  # constructor
  def __init__(self):
    self.window = sg.Window("RIVET BCI", self.layout, margins=(20, 20), finalize=True)
    self.base_model_loaded = False
    self.game_connected = False
    self.headset_connected = False
    self.model_trained = False

    self.model = None
    self.game = None
    self.eeg = None

    self.to_classify = deque(maxlen=500)
    self.target = None

    self.gui_updates = {}
    self.loading = set()

  def run(self):
    # ThreadedModel(parent=self).start()
    Thread(target=self.train_model, args=(True,)).start()
    self.loop()

  def loop(self):
    """
    The main application loop and event handler.
    """
    while True:
      if self.gui_updates:
        self.update_gui()

      event, values = self.window.read(100)

      if event == "btn_con_game":
        Thread(target=self.connect_game, daemon=True).start()
      elif event == "btn_con_headset":
        Thread(target=self.connect_headset, daemon=True).start()
      elif event == "btn_train_model":
        Thread(target=self.train_model, daemon=True).start()
      elif event == "btn_finalize":
        Thread(target=self.finalize, daemon=True).start()
      
      to_update = self.loading.copy()
      for update in to_update:
        self.window.Element(f'{update}_loading').UpdateAnimation('assets/loading.gif')

      # End program if user closes window
      if event == sg.WIN_CLOSED:
        break

    self.window.close()

  def connect_game(self):
    """
    Connect the BCI application to the ForestShepherd game.
    """
    if 'game_con' in self.loading:
      return

    self.queue_gui_update('game_con_status', {'visible': False})
    self.queue_gui_update('btn_con_game', {'text': 'Connecting...'})
    self.queue_gui_update('game_con_loading', {'visible': True})
    self.loading.add('game_con')

    result = game_connect() if self.game is None else False
    if result is not False:
      self.game = result
      # Thread(target=self.receive).start()

      self.queue_gui_update('game_con_status', {'value': u'\u2713', 'text_color': 'green', 'visible': True})
      self.queue_gui_update('btn_con_game', {'text': 'Disconnect game'})
    else:
      self.game = game_disconnect(self.game) if self.game is not None else None
      self.queue_gui_update('game_con_status', {'value': u'\u2717', 'text_color': 'red', 'visible': True})
      self.queue_gui_update('btn_con_game', {'text': 'Connect game'})

    self.queue_gui_update('game_con_loading', {'visible': False})
    self.loading.remove('game_con')
    
  def connect_headset(self):
    """
    Connect the BCI application to the EEG headset hardware.
    """
    if 'headset_con' in self.loading:
      return

    self.queue_gui_update('headset_con_status', {'visible': False})
    self.queue_gui_update('btn_con_headset', {'text': 'Connecting...'})
    self.queue_gui_update('headset_con_loading', {'visible': True})
    self.loading.add('headset_con') 

    self.eeg = lsl_connect() if self.eeg is None else None
    if self.eeg is not None:
      # Thread(target=self.stream).start()

      self.queue_gui_update('headset_con_status', {'value': u'\u2713', 'text_color': 'green', 'visible': True})
      self.queue_gui_update('btn_con_headset', {'text': 'Disconnect headset'})
      self.queue_gui_update('headset_con_loading', {'visible': False})
    else:
      self.queue_gui_update('headset_con_status', {'value': u'\u2717', 'text_color': 'red', 'visible': True})
      self.queue_gui_update('btn_con_headset', {'text': 'Connect headset'})
      self.queue_gui_update('headset_con_loading', {'visible': False})

    self.loading.remove('headset_con')

  def train_model(self, initial=False):
    """
    Select a file containing transfer learning information and train the model
    to be used as the BCI classifier. If the model is already trained then this
    method doubles as a model reset to base weights.
    """
    if len(self.loading.intersection({'model_train', 'base_model'})) > 0:
      return

    train_files = []
    if not self.model_trained and not initial:
      train_files = list(askopenfilenames())
      if len(train_files) is 0:
        return

      self.queue_gui_update('model_train_status', {'visible': False})
      self.queue_gui_update('btn_train_model', {'text': 'Training...'})
      self.queue_gui_update('model_train_loading', {'visible': True})
      self.loading.add('model_train')
    else:
      self.queue_gui_update('base_model_status', {'visible': False})
      self.queue_gui_update('btn_train_model', {'text': 'Loading base model...'})
      self.queue_gui_update('base_model_loading', {'visible': True})
      self.loading.add('base_model')


    self.model, is_base = get_model(train_files)
    if is_base or is_base is None:
      self.base_model_loaded = True
      self.model_trained = False
      self.queue_gui_update('base_model_status', {'value': u'\u2713', 'text_color': 'green', 'visible': True})
      self.queue_gui_update('base_model_loading', {'visible': False})

      self.queue_gui_update('model_train_status', {'value': u'\u2717', 'text_color': 'red', 'visible': True})
      self.queue_gui_update('btn_train_model', {'text': 'Train transfer model'})
      self.queue_gui_update('model_train_loading', {'visible': False})

      if is_base:
        self.loading.remove('base_model')
    else:
      self.model_trained = True
      self.queue_gui_update('model_train_status', {'value': u'\u2713', 'text_color': 'green', 'visible': True})
      self.queue_gui_update('btn_train_model', {'text': 'Reset model'})
      self.queue_gui_update('model_train_loading', {'visible': False})
      self.loading.remove('model_train')

    self.model._make_predict_function()

  def finalize(self):
    print("finalize")
    if self.game is not None:
      print("starting stream and socket threads")
      Thread(target=self.stream, daemon=True).start()
      Thread(target=self.receive, daemon=True).start()

    

  def receive(self):
    print("starting receive thread")
    while self.game is not None:
      print('waiting for target')
      self.target = int.from_bytes(self.game.recv(1), byteorder="big")
      print(self.target)

  def stream(self):
    print("starting debug stream thread")

    debug_index = 0
    while True:
      while self.target is None:
        pass

      sleep(2.5)
      # grab debug epoch
      epoch = pilot[debug_index]
      target = labels[debug_index]

      to_classify = np.stack([epoch])
      print(to_classify.shape)

      # reshape to [epochs (1), kernels (1), channels (?), samples (1000)] 
      probs = self.model.predict(to_classify.reshape(to_classify.shape[0], 1, to_classify.shape[1], to_classify.shape[2]))
      probs = np.sum(probs, axis=0) / 1
      print(probs)

      result = np.where(probs > 0.66)
      print("debug target:", target, f"({label_map[target]})")
      # self.game.sendall(bytes([self.target + 1]))
      if len(result[0]) == 0:
        # send unknown
        print('unknown')
        self.game.sendall(bytes([25]))
      else:
        # send index of result
        print('classified:', result[0][0], f"({label_map[result[0][0]]})")
        self.game.sendall(bytes([result[0][0] + 1]))

      self.target = None
      debug_index += 1


  def queue_gui_update(self, element_key, update_dict):
    """
    Adds a single GUI update to the update dictionary
    """
    self.gui_updates[element_key] = update_dict

  def update_gui(self):
    """
    Performs all queued updates and resets the update dictionary
    """
    for where, updates in self.gui_updates.items():
      self.window[where].update(**updates)
    self.gui_updates = {}



if __name__ == "__main__":
  app = App()
  app.run()