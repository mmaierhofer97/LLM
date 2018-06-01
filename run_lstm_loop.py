import os
cwd = os.path.join(os.getcwd(), "LSTM_RUN.py")
for i in range(10):
    print(i)
    os.system('{} {}'.format('python3', cwd))
