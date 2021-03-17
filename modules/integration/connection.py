from socket import socket, AF_INET, SOCK_STREAM, SHUT_RDWR

HOST = '127.0.0.1'    # The server's hostname or IP address
PORT = 65432          # The port used by the server

def connect():
  connection = socket(AF_INET, SOCK_STREAM)
  try:
    connection.connect((HOST, PORT))
  except:
    print("not connected")
    return False

  print("connected", connection)
  return connection

def disconnect(socket: socket):
  socket.shutdown(SHUT_RDWR)
  socket.close()

  return None         # Returns None so self.socket can be easily set to None in one line