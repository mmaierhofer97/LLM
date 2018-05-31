import os
cwd = os.path.join(os.getcwd(), "llm_run.py")
for i in range(10):
    print(i)
    os.system('{} {}'.format('sudo python3', cwd))
