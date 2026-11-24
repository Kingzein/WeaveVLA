"""Task-local resident processes, inheriting exactly the selected idle GPU."""
import os,time,subprocess
from pathlib import Path
from multiprocessing.connection import Client
ROOT=Path(os.environ['WEAVEVLA_RUNTIME_ROOT'])
WORKER=Path(__file__).with_name('resident_worker.py')
class PerceptionPool:
    def __init__(self,output):self.output=Path(output);self.children={};self.sockets={}
    def start(self):
        try:
            for name,python in [('sam3',os.environ['SAM3_PYTHON']),
                                ('yoloe',os.environ['YOLO_PYTHON'])]:
                socket=Path(f'/tmp/weavevla-memory-perception-{os.getpid()}-{name}.sock')
                socket.unlink(missing_ok=True)
                # No unrelated cache routing/capacity sitecustomize injection.
                env=os.environ.copy()
                env.pop('GRAPH_CACHE_GPU_CAP_GIB',None)
                with open(self.output/f'perception_{name}.log','a') as log:
                    proc=subprocess.Popen([python,'-u',str(WORKER),name,'--socket',str(socket),
                        '--parent-pid',str(os.getpid())],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                self.children[name]=proc;self.sockets[name]=socket;deadline=time.monotonic()+300
                while not socket.exists():
                    if proc.poll() is not None:raise RuntimeError(f'{name} failed; see perception log')
                    if time.monotonic()>deadline:raise TimeoutError(f'{name} loading timeout')
                    time.sleep(.2)
        except BaseException:self.close();raise
    def run(self,stage,folder):
        name={'general_detect':'yoloe','segment':'sam3'}[stage]
        if self.children[name].poll() is not None:raise RuntimeError(f'{name} exited')
        with Client(str(self.sockets[name]),family='AF_UNIX',authkey=b'weavevla-yoloe-v2-resident') as conn:
            conn.send(dict(folder=str(folder),phase=None))
            if not conn.poll(300):raise TimeoutError(stage)
            response=conn.recv()
            if not response['ok']:raise RuntimeError(response['error'])
    def close(self):
        for proc in self.children.values():
            if proc.poll() is None:proc.terminate()
        for proc in self.children.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for socket in self.sockets.values():socket.unlink(missing_ok=True)
