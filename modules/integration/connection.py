from socket import socket, AF_INET, SOCK_STREAM, SHUT_RDWR
from pylsl import StreamInlet, resolve_byprop

HOST = '127.0.0.1'    # The server's hostname or IP address
PORT = 65432          # The port used by the server

def game_connect():
  connection = socket(AF_INET, SOCK_STREAM)
  try:
    connection.connect((HOST, PORT))
  except:
    print("not connected")
    return False

  print("connected", connection)
  return connection

def game_disconnect(socket: socket):
  socket.shutdown(SHUT_RDWR)
  socket.close()

  return None         # Returns None so self.socket can be easily set to None in one line


def lsl_connect():
  # first resolve an EEG stream on the lab network
  print("looking for an EEG stream...")
  streams = resolve_byprop('type', 'EEG', timeout=5)

  # create a new inlet to read from the stream
  if len(streams) > 0:
    inlet = StreamInlet(streams[0])
    print('found EEG stream', inlet)
    return inlet
  else:
     return None

def lsl_disconnect():
  pass